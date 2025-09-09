const std = @import("std");
const ssim = @import("ssimulacra2.zig");
const io = @import("io.zig");
const print = std.debug.print;
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("jpeglib.h");
});

const VERSION = @import("build_opts").version;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);
    while (args_iter.next()) |a| try args.append(allocator, a);

    var json_output = false;
    var positional: [2]?[]const u8 = .{ null, null };
    var pos_index: usize = 0;

    var show_help = false;
    var show_version = false;

    for (args.items[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            show_help = true;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            show_version = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
        } else {
            if (pos_index >= 2)
                return usageExtra("Too many positional arguments provided.");
            positional[pos_index] = arg;
            pos_index += 1;
        }
    }

    if (show_help) return usage();
    if (show_version) return printVersion();

    if (pos_index != 2)
        return usageExtra("Two image paths required: reference distorted");

    const ref_path = positional[0].?;
    const dist_path = positional[1].?;

    var ref_image = try io.loadImage(allocator, ref_path);
    defer ref_image.deinit(allocator);

    var dist_image = try io.loadImage(allocator, dist_path);
    defer dist_image.deinit(allocator);

    // Enforce matching original dimensions
    if (ref_image.width != dist_image.width or ref_image.height != dist_image.height)
        return fail("Input images must have identical dimensions (got {d}x{d} vs {d}x{d})", .{ ref_image.width, ref_image.height, dist_image.width, dist_image.height }, 2);

    // Convert both to 3-channel RGB ignoring alpha (if present)
    const ref_has_alpha: bool = ref_image.channels != 3;
    const ref_rgb: []u8 = if (ref_has_alpha) try io.toRGB8(allocator, ref_image) else ref_image.data;
    defer if (ref_has_alpha) allocator.free(ref_rgb);

    const dst_has_alpha: bool = dist_image.channels != 3;
    const dist_rgb: []u8 = if (dst_has_alpha) try io.toRGB8(allocator, dist_image) else dist_image.data;
    defer if (dst_has_alpha) allocator.free(dist_rgb);

    const score = ssim.computeSsimu2(
        allocator,
        ref_rgb,
        dist_rgb,
        @intCast(ref_image.width),
        @intCast(ref_image.height),
        3,
    ) catch |e| {
        switch (e) {
            ssim.Ssimu2Error.InvalidChannelCount => {
                return fail("Invalid channel count encountered.", .{}, 3);
            },
            ssim.Ssimu2Error.OutOfMemory => {
                return fail("Out of memory allocating working buffers.", .{}, 3);
            },
        }
    };

    if (json_output) {
        print(
            "{{\"metric\":\"SSIMULACRA2\",\"score\":{d:.8}}}\n",
            .{score},
        );
    } else print("{d:.8}\n", .{score});
    return;
}

fn usage() void {
    print("\x1b[34mfssimu2\x1b[0m | {s}\n\n", .{VERSION});
    print(
        \\usage:
        \\  fssimu2 [--json] reference.(png|pam|jpg|jpeg) distorted.(png|pam|jpg|jpeg)
        \\
        \\options:
        \\  --json          output result as json
        \\  -h, --help      show this help
        \\  -v, --version   show version information
    , .{});
    print("\n\n\x1b[37m8-bit sRGB PNG, PAM, or JPEG expected (RGB[A] or GRAYSCALE[+ALPHA])\x1b[0m\n", .{});
}

fn printVersion() void {
    const jpeg_version = c.LIBJPEG_TURBO_VERSION_NUMBER;
    const major = jpeg_version / 1_000_000;
    const minor = (jpeg_version / 1_000) % 1_000;
    const patch = jpeg_version % 1_000;
    const jpeg_simd: bool = c.WITH_SIMD != 0;
    print("fssimu2 {s}\n", .{VERSION});
    print("libjpeg-turbo {d}.{d}.{d} [simd: {}]\n", .{ major, minor, patch, jpeg_simd });
}

fn usageExtra(msg: []const u8) void {
    print("Error: {s}\n\n", .{msg});
    usage();
}

fn fail(comptime fmt: []const u8, args: anytype, code: u8) void {
    print("Error: " ++ fmt ++ "\n", args);
    std.process.exit(code);
}
