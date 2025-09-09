const std = @import("std");
const c = @cImport({
    @cInclude("third-party/libspng/spng.h");
    @cInclude("jpeglib.h");
});

// Generic in-memory image representation used by the metric pipeline.
pub const Image = struct {
    width: usize,
    height: usize,
    channels: u8, // 1=Gray,2=GrayA,3=RGB,4=RGBA
    data: []u8, // interleaved, row-major

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

pub fn loadJPEG(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});

    const file_ptr = c.fdopen(file.handle, "rb");
    if (file_ptr == null) {
        file.close();
        return error.FailedToOpenFile;
    }
    defer _ = c.fclose(file_ptr);

    var cinfo: c.jpeg_decompress_struct = undefined;
    var jerr: c.jpeg_error_mgr = undefined;

    cinfo.err = c.jpeg_std_error(&jerr);
    c.jpeg_create_decompress(&cinfo);
    defer c.jpeg_destroy_decompress(&cinfo);

    c.jpeg_stdio_src(&cinfo, file_ptr);

    if (c.jpeg_read_header(&cinfo, c.TRUE) != c.JPEG_HEADER_OK)
        return error.InvalidJPEGHeader;

    if (cinfo.num_components == 1)
        cinfo.out_color_space = c.JCS_GRAYSCALE
    else
        cinfo.out_color_space = c.JCS_RGB;

    if (c.jpeg_start_decompress(&cinfo) != c.TRUE) {
        return error.JPEGDecompressFailed;
    }

    const width: usize = @intCast(cinfo.output_width);
    const height: usize = @intCast(cinfo.output_height);
    const channels: usize = @intCast(cinfo.output_components);

    const row_stride: usize = width * channels;
    const out_buf: []u8 = try allocator.alloc(u8, height * row_stride);
    errdefer allocator.free(out_buf);

    const row_buf = try allocator.alloc(u8, row_stride);
    defer allocator.free(row_buf);

    for (0..height) |y| {
        var row_pointers: [1][*c]u8 = .{row_buf.ptr};
        if (c.jpeg_read_scanlines(&cinfo, &row_pointers, 1) != 1)
            return error.JPEGReadScanlinesFailed;
        @memcpy(out_buf[y * row_stride .. (y + 1) * row_stride], row_buf);
    }

    if (c.jpeg_finish_decompress(&cinfo) != c.TRUE)
        return error.JPEGFinishDecompressFailed;

    return .{
        .width = @intCast(width),
        .height = @intCast(height),
        .channels = @intCast(channels),
        .data = out_buf,
    };
}

pub fn loadPNG(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const size = try file.getEndPos();
    const buf = try allocator.alloc(u8, size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);

    const ctx = c.spng_ctx_new(0);
    if (ctx == null) return error.FailedCreateContext;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_png_buffer(ctx, buf.ptr, buf.len) != 0)
        return error.SetBufferFailed;

    var ihdr: c.struct_spng_ihdr = undefined;
    if (c.spng_get_ihdr(ctx, &ihdr) != 0)
        return error.GetHeaderFailed;

    // prefer RGB8 (ignore alpha), but if alpha, decode RGBA8 & strip later.
    const fmt: c_int = switch (ihdr.color_type) {
        c.SPNG_COLOR_TYPE_TRUECOLOR => c.SPNG_FMT_RGB8,
        c.SPNG_COLOR_TYPE_TRUECOLOR_ALPHA => c.SPNG_FMT_RGBA8,
        c.SPNG_COLOR_TYPE_GRAYSCALE => c.SPNG_FMT_RGBA8, // libspng expands to RGBA; we'll drop A later
        c.SPNG_COLOR_TYPE_GRAYSCALE_ALPHA => c.SPNG_FMT_RGBA8,
        c.SPNG_COLOR_TYPE_INDEXED => c.SPNG_FMT_RGBA8,
        else => c.SPNG_FMT_RGBA8,
    };

    var out_size: usize = 0;
    if (c.spng_decoded_image_size(ctx, fmt, &out_size) != 0) return error.ImageSizeFailed;

    const out_buf = try allocator.alloc(u8, out_size);
    errdefer allocator.free(out_buf);

    if (c.spng_decode_image(ctx, out_buf.ptr, out_size, fmt, 0) != 0) return error.DecodeFailed;

    const channels: u8 = switch (fmt) {
        c.SPNG_FMT_RGB8 => 3,
        else => 4,
    };

    return .{
        .width = ihdr.width,
        .height = ihdr.height,
        .channels = channels,
        .data = out_buf,
    };
}

pub fn loadPAM(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buf = try allocator.alloc(u8, file_size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);

    if (buf.len < 3 or !std.mem.startsWith(u8, buf, "P7")) return error.NotAPamFile;

    // Find header end. Prefer explicit ENDHDR marker; else look for double newline.
    const endhdr_explicit = std.mem.indexOf(u8, buf, "ENDHDR\n");
    var header_end_index: ?usize = null;
    if (endhdr_explicit) |i| {
        header_end_index = i + 7; // include terminator
    } else {
        // Look for first occurrence of "\n\n" (empty line). PAM spec mandates ENDHDR
        // but some generators may still use empty line.
        const empty_line = std.mem.indexOf(u8, buf, "\n\n");
        if (empty_line) |i| header_end_index = i + 2;
    }
    if (header_end_index == null) return error.HeaderNotFound;
    const header_end = header_end_index.?;

    const header = buf[0..header_end];

    var width: usize = 0;
    var height: usize = 0;
    var depth: usize = 0;
    var maxval: usize = 0;
    var tuple_type: []const u8 = "UNSPECIFIED";

    var line_it = std.mem.tokenizeAny(u8, header, "\r\n");
    while (line_it.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '#') continue; // comment
        if (std.mem.startsWith(u8, line, "WIDTH")) {
            var it = std.mem.tokenizeAny(u8, line[5..], " \t");
            if (it.next()) |v| width = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.startsWith(u8, line, "HEIGHT")) {
            var it = std.mem.tokenizeAny(u8, line[6..], " \t");
            if (it.next()) |v| height = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.startsWith(u8, line, "DEPTH")) {
            var it = std.mem.tokenizeAny(u8, line[5..], " \t");
            if (it.next()) |v| depth = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.startsWith(u8, line, "MAXVAL")) {
            var it = std.mem.tokenizeAny(u8, line[6..], " \t");
            if (it.next()) |v| maxval = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.startsWith(u8, line, "TUPLTYPE")) {
            var it = std.mem.tokenizeAny(u8, line[8..], " \t");
            if (it.next()) |v| tuple_type = v;
        } else if (std.mem.eql(u8, line, "ENDHDR")) {
            break;
        }
    }

    if (width == 0 or height == 0 or depth == 0 or maxval == 0)
        return error.InvalidPamDimensions;
    if (maxval != 255) return error.UnsupportedPamMaxVal;
    if (depth != 1 and depth != 2 and depth != 3 and depth != 4)
        return error.UnsupportedPamDepth;

    var channels: u8 = @intCast(depth);
    if (std.ascii.eqlIgnoreCase(tuple_type, "GRAYSCALE")) {
        if (depth != 1) return error.PamTupleMismatch;
        channels = 1;
    } else if (std.ascii.eqlIgnoreCase(tuple_type, "GRAYSCALE_ALPHA")) {
        if (depth != 2) return error.PamTupleMismatch;
        channels = 2;
    } else if (std.ascii.eqlIgnoreCase(tuple_type, "RGB")) {
        if (depth != 3) return error.PamTupleMismatch;
        channels = 3;
    } else if (std.ascii.eqlIgnoreCase(tuple_type, "RGB_ALPHA")) {
        if (depth != 4) return error.PamTupleMismatch;
        channels = 4;
    } else if (std.ascii.eqlIgnoreCase(tuple_type, "BLACKANDWHITE")) {
        // binary (maxval should be 1) - not supporting
        return error.UnsupportedPamTuple;
    }

    const pixel_count = width * height;
    const data_size = pixel_count * channels;
    if (header_end + data_size > buf.len) return error.InsufficientDataInFile;

    const out = try allocator.alloc(u8, pixel_count * channels);
    errdefer allocator.free(out);
    @memcpy(out, buf[header_end .. header_end + data_size]);

    return .{
        .width = width,
        .height = height,
        .channels = channels,
        .data = out,
    };
}

pub fn loadImage(allocator: std.mem.Allocator, path: []const u8) !Image {
    if (hasExtension(path, ".png"))
        return loadPNG(allocator, path);
    if (hasExtension(path, ".pam"))
        return loadPAM(allocator, path);
    if (hasExtension(path, ".jpg") or hasExtension(path, ".jpeg"))
        return loadJPEG(allocator, path);

    return loadPNG(allocator, path) catch loadPAM(allocator, path) catch loadJPEG(allocator, path);
}

fn hasExtension(path: []const u8, ext: []const u8) bool {
    if (path.len < ext.len) return false;
    const tail = path[path.len - ext.len ..];
    return std.ascii.eqlIgnoreCase(tail, ext);
}

pub fn toRGB8(allocator: std.mem.Allocator, img: Image) ![]u8 {
    const pixels = img.width * img.height;
    const rgb = try allocator.alloc(u8, pixels * 3);
    switch (img.channels) {
        3 => {
            // direct copy
            var i: usize = 0;
            while (i < pixels) : (i += 1) {
                rgb[i * 3 + 0] = img.data[i * 3 + 0];
                rgb[i * 3 + 1] = img.data[i * 3 + 1];
                rgb[i * 3 + 2] = img.data[i * 3 + 2];
            }
        },
        4 => {
            var i: usize = 0;
            while (i < pixels) : (i += 1) {
                rgb[i * 3 + 0] = img.data[i * 4 + 0];
                rgb[i * 3 + 1] = img.data[i * 4 + 1];
                rgb[i * 3 + 2] = img.data[i * 4 + 2];
            }
        },
        1 => {
            // replicate grayscale channel
            var i: usize = 0;
            while (i < pixels) : (i += 1) {
                const g = img.data[i];
                rgb[i * 3 + 0] = g;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = g;
            }
        },
        2 => {
            // grayscale + alpha -> ignore alpha
            var i: usize = 0;
            while (i < pixels) : (i += 1) {
                const g = img.data[i * 2 + 0];
                rgb[i * 3 + 0] = g;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = g;
            }
        },
        else => return error.UnsupportedChannelCount,
    }
    return rgb;
}
