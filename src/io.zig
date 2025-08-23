const std = @import("std");
const c = @cImport({
    @cInclude("third-party/libspng/spng.h");
});

const PNGImage = struct {
    width: usize,
    height: usize,
    channels: u8,
    data: []u8, // interleaved

    pub fn deinit(self: *PNGImage, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

pub fn loadPNG(allocator: std.mem.Allocator, path: []const u8) !PNGImage {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const size = try file.getEndPos();
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    _ = try file.readAll(buf);

    const ctx = c.spng_ctx_new(0);
    if (ctx == null) return error.FailedCreateContext;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_png_buffer(ctx, buf.ptr, buf.len) != 0) return error.SetBufferFailed;

    var ihdr: c.struct_spng_ihdr = undefined;
    if (c.spng_get_ihdr(ctx, &ihdr) != 0) return error.GetHeaderFailed;

    // prefer RGB8 (ignore alpha), but if alpha, decode RGBA8 & strip later.
    const fmt: c_int = switch (ihdr.color_type) {
        c.SPNG_COLOR_TYPE_TRUECOLOR => c.SPNG_FMT_RGB8,
        c.SPNG_COLOR_TYPE_TRUECOLOR_ALPHA => c.SPNG_FMT_RGBA8,
        c.SPNG_COLOR_TYPE_GRAYSCALE => c.SPNG_FMT_RGBA8, // force expand later (libspng can do transforms)
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

    return PNGImage{
        .width = ihdr.width,
        .height = ihdr.height,
        .channels = channels,
        .data = out_buf,
    };
}

pub fn toRGB8(allocator: std.mem.Allocator, img: PNGImage) ![]u8 {
    const pixels = img.width * img.height;
    const rgb = try allocator.alloc(u8, pixels * 3);
    if (img.channels == 3) {
        for (0..pixels) |i| {
            rgb[i * 3 + 0] = img.data[i * 3 + 0];
            rgb[i * 3 + 1] = img.data[i * 3 + 1];
            rgb[i * 3 + 2] = img.data[i * 3 + 2];
        }
    } else { // RGBA - drop alpha
        for (0..pixels) |i| {
            rgb[i * 3 + 0] = img.data[i * 4 + 0];
            rgb[i * 3 + 1] = img.data[i * 4 + 1];
            rgb[i * 3 + 2] = img.data[i * 4 + 2];
        }
    }
    return rgb;
}

pub fn padWidth(
    allocator: std.mem.Allocator,
    original: []const u8,
    width: usize,
    height: usize,
    new_width: usize,
) ![]u8 {
    if (new_width == width) return @constCast(original);
    const channels: usize = 3;
    const old_row_bytes = width * channels;
    const new_row_bytes = new_width * channels;
    var out = try allocator.alloc(u8, height * new_row_bytes);

    for (0..height) |y| {
        const src_row = original[y * old_row_bytes .. (y + 1) * old_row_bytes];
        const dst_row = out[y * new_row_bytes .. (y + 1) * new_row_bytes];
        @memcpy(dst_row[0..old_row_bytes], src_row);
        const last_px_offset = old_row_bytes - channels;
        var x_bytes: usize = old_row_bytes;
        while (x_bytes < new_row_bytes) : (x_bytes += channels) {
            @memcpy(dst_row[x_bytes .. x_bytes + channels], src_row[last_px_offset .. last_px_offset + channels]);
        }
    }
    return out;
}
