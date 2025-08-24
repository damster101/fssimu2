const std = @import("std");
const ssim = @import("ssimulacra2.zig");
const io = @import("io.zig");
const print = std.debug.print;

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

    if (args.items.len < 3)
        return usage();

    var json_output = false;
    var positional: [2]?[]const u8 = .{ null, null };
    var pos_index: usize = 0;

    for (args.items[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return usage();
        } else if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
        } else {
            if (pos_index >= 2)
                return usageExtra("Too many positional arguments provided.");
            positional[pos_index] = arg;
            pos_index += 1;
        }
    }

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
    const ref_rgb_allocated: bool = ref_image.channels != 3;
    const ref_rgb = if (ref_rgb_allocated) try io.toRGB8(allocator, ref_image) else ref_image.data;
    defer if (ref_rgb_allocated) allocator.free(ref_rgb);

    const dist_rgb_allocated: bool = dist_image.channels != 3;
    const dist_rgb = if (dist_rgb_allocated) try io.toRGB8(allocator, dist_image) else dist_image.data;
    defer if (dist_rgb_allocated) allocator.free(dist_rgb);

    var width = ref_image.width;
    const height = ref_image.height;

    // If width not multiple of 16, pad both horizontally (replicate last pixel)
    var padded_ref = ref_rgb;
    var padded_dist = dist_rgb;
    if (width % 16 != 0) {
        const padded_width = ((width + 15) / 16) * 16;
        padded_ref = try io.padWidth(allocator, ref_rgb, width, height, padded_width);
        defer if (padded_ref.ptr != ref_rgb.ptr) allocator.free(ref_rgb);
        padded_dist = try io.padWidth(allocator, dist_rgb, width, height, padded_width);
        defer if (padded_dist.ptr != dist_rgb.ptr) allocator.free(dist_rgb);
        width = padded_width;
    }

    const score = ssim.computeSSIMULACRA2(
        allocator,
        padded_ref,
        padded_dist,
        @intCast(width),
        @intCast(height),
        3,
    ) catch |e| {
        switch (e) {
            ssim.Ssimu2Error.WidthNotMultipleOf16 => {
                return fail("Width not multiple of 16 even after padding attempt.", .{}, 3);
            },
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
        \\  fssimu2 [--json] reference.(png|pam) distorted.(png|pam)
        \\
        \\options:
        \\  --json          output result as json
        \\  -h, --help      show this help
    , .{});
    print("\n\n\x1b[37m8-bit sRGB PNG or PAM expected (RGB[A] or GRAYSCALE[+ALPHA])\x1b[0m\n", .{});
}

fn usageExtra(msg: []const u8) void {
    print("Error: {s}\n\n", .{msg});
    usage();
}

fn fail(comptime fmt: []const u8, args: anytype, code: u8) void {
    print("Error: " ++ fmt ++ "\n", args);
    std.process.exit(code);
}
