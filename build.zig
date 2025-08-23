const std = @import("std");

fn getVersionString(b: *std.Build) ![]const u8 {
    const allocator = b.allocator;
    const command = [_][]const u8{ "git", "describe", "--tags", "--always" };
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &command,
    }) catch |err| {
        std.log.warn("Failed to get git version: {s}", .{@errorName(err)});
        return "unknown";
    };
    if (result.term.Exited != 0)
        return "unknown";
    const version = std.mem.trimRight(u8, result.stdout, "\r\n");
    return allocator.dupe(u8, version);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const options = b.addOptions();
    const version = getVersionString(b) catch "unknown";
    options.addOption([]const u8, "version", version);

    // libspng
    const spng = b.addLibrary(.{
        .name = "spng",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    const spng_sources = [_][]const u8{
        "third-party/libspng/spng.c",
        "third-party/libminiz/miniz.c",
    };
    spng.linkLibC();
    spng.linkSystemLibrary("m");
    spng.addCSourceFiles(.{ .files = &spng_sources });
    spng.addIncludePath(b.path("third-party/"));

    // ssimu2
    const bin = b.addExecutable(.{
        .name = "ssimu2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    bin.root_module.addOptions("build_opts", options);
    bin.addIncludePath(b.path("."));
    bin.linkLibC();
    bin.linkLibrary(spng);

    // Install step
    b.installArtifact(bin);
}
