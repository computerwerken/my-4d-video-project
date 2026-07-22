# Locate system-installed dependencies instead of hardcoding absolute paths.
#
# The upstream WORKSPACE hardcoded paths like /usr/local/Cellar/opencv/4.11.0_1
# (macOS) and /usr/local (Linux). Both are wrong on common setups:
#
#   macOS
#     1. Apple Silicon installs Homebrew under /opt/homebrew, not /usr/local, so
#        every Cellar path is wrong on an M-series Mac.
#     2. The version component ("4.11.0_1") changes on any `brew upgrade`, so
#        the build breaks on a schedule set by Homebrew.
#     3. Some formulae are versioned: plain `opencv` is not a keg, it is
#        `opencv@4`. Getting this wrong yields a confusing "no such package".
#
#   Linux
#     4. `apt install libopencv-dev` puts headers in /usr, not /usr/local, and
#        libraries in a triplet dir (lib/x86_64-linux-gnu, lib/aarch64-linux-gnu)
#        rather than lib/.
#     5. Ubuntu does not ship the contrib module libopencv_xfeatures2d, so
#        linking against it fails on a stock apt install.
#
# Both rules below probe the machine and fail with an actionable message naming
# the exact install command when something is missing. Bazel evaluates
# repository rules lazily, so the macOS rules never execute on Linux and vice
# versa.

# ==============================================================================
# macOS: Homebrew
# ==============================================================================

# Searched in order if `brew` is not already on PATH.
_BREW_CANDIDATES = [
    "/opt/homebrew/bin/brew",               # Apple Silicon default
    "/usr/local/bin/brew",                  # Intel default
    "/home/linuxbrew/.linuxbrew/bin/brew",  # Linuxbrew, for completeness
]

# Subdirectories exposed to the generated BUILD file. include/ and lib/ are what
# every consumer actually globs; the rest are linked when present because they
# cost nothing and may save a future debugging session.
_BREW_SUBDIRS = ["include", "lib", "bin", "share", "libexec"]

def _find_brew(rctx):
    on_path = rctx.which("brew")
    if on_path:
        return on_path
    for candidate in _BREW_CANDIDATES:
        if rctx.path(candidate).exists:
            return rctx.path(candidate)
    return None

def _homebrew_package_impl(rctx):
    formula = rctx.attr.formula
    brew = _find_brew(rctx)
    if brew == None:
        fail(
            "Could not find `brew` while resolving @%s (formula '%s').\n" % (rctx.name, formula) +
            "Install Homebrew from https://brew.sh, then:\n" +
            "    brew install %s" % formula,
        )

    result = rctx.execute([str(brew), "--prefix", formula])
    if result.return_code != 0:
        fail(
            "`brew --prefix %s` failed while resolving @%s.\n" % (formula, rctx.name) +
            "This usually means the formula is not installed. Try:\n" +
            "    brew install %s\n" % formula +
            "brew said: %s" % result.stderr.strip(),
        )

    prefix_str = result.stdout.strip()
    if not prefix_str:
        fail("`brew --prefix %s` returned an empty path (@%s)." % (formula, rctx.name))

    prefix = rctx.path(prefix_str)
    if not prefix.exists:
        fail(
            "Homebrew reported '%s' for formula '%s', but that path does not exist.\n" % (prefix_str, formula) +
            "The formula is probably registered but not installed. Try:\n" +
            "    brew install %s" % formula,
        )

    linked = []
    for sub in _BREW_SUBDIRS:
        child = prefix.get_child(sub)
        if child.exists:
            rctx.symlink(child, sub)
            linked.append(sub)

    # A keg with neither headers nor libraries cannot satisfy any consumer here,
    # and the resulting error (an empty glob) would be opaque. Fail loudly.
    if "include" not in linked and "lib" not in linked:
        fail(
            "Homebrew formula '%s' resolved to '%s' but has no include/ or lib/ " % (formula, prefix_str) +
            "directory, so @%s cannot provide headers or libraries.\n" % rctx.name +
            "Try: brew reinstall %s" % formula,
        )

    rctx.file("BUILD", rctx.attr.build_file_content, executable = False)

homebrew_package = repository_rule(
    implementation = _homebrew_package_impl,
    doc = "A local repository backed by `brew --prefix <formula>`.",
    attrs = {
        "formula": attr.string(
            mandatory = True,
            doc = "Homebrew formula name, e.g. 'opencv@4' or 'ceres-solver'.",
        ),
        "build_file_content": attr.string(
            mandatory = True,
            doc = "BUILD file contents; globs may use include/ and lib/ as usual.",
        ),
    },
    # local: re-resolve on each build so a `brew upgrade` is picked up rather
    # than silently serving a stale cached path.
    local = True,
    configure = True,
)

# ==============================================================================
# Linux: system OpenCV (apt or source build)
# ==============================================================================

# /usr/local first so a deliberate source build wins over an apt install, which
# preserves upstream's original intent; /usr covers `apt install libopencv-dev`.
_OPENCV_PREFIXES = ["/usr/local", "/usr"]

# Debian/Ubuntu use multiarch triplet dirs; RPM distros use lib64; source builds
# use plain lib.
_OPENCV_LIBDIRS = [
    "lib/x86_64-linux-gnu",
    "lib/aarch64-linux-gnu",
    "lib64",
    "lib",
]

# Linked only if actually present on the machine. xfeatures2d is a contrib
# module Ubuntu does not ship; the LDI pipeline never calls it (only the
# SfM/Gaussian side references it), so its absence must not break the build,
# while its presence should still be linked for the targets that do want it.
_OPENCV_LINK_CANDIDATES = [
    "opencv_core",
    "opencv_features2d",
    "opencv_flann",
    "opencv_imgproc",
    "opencv_calib3d",
    "opencv_xfeatures2d",
    "opencv_highgui",
    "opencv_imgcodecs",
]

_OPENCV_BUILD_TEMPLATE = """# Generated by local_opencv in system_deps.bzl.
# Resolved prefix: {prefix}
# Resolved libdir: {libdir}
cc_library(
  name = "opencv",
  srcs = glob(["lib/libopencv_*.so*"]),
  hdrs = glob([
    "include/opencv4/opencv2/**/*.h",
    "include/opencv4/opencv2/**/*.hpp",
    "include/opencv4/opencv2/*.hpp"
  ]),
  includes = ["include/opencv4"],
  strip_include_prefix = "include/opencv4",
  deps = ["@glib//:glib"],
  copts = ["-O3"],
  linkopts = [{linkopts}],
  visibility = ["//visibility:public"],
  linkstatic = 1,
)
"""

def _has_glob(rctx, pattern):
    """True if the shell glob matches at least one file."""
    res = rctx.execute(["sh", "-c", "ls %s >/dev/null 2>&1" % pattern])
    return res.return_code == 0

def _local_opencv_impl(rctx):
    prefix = None
    for candidate in rctx.attr.prefixes:
        if rctx.path(candidate + "/include/opencv4/opencv2/core.hpp").exists:
            prefix = candidate
            break
    if prefix == None:
        fail(
            "Could not find OpenCV 4 headers while resolving @%s.\n" % rctx.name +
            "Looked for include/opencv4/opencv2/core.hpp under: %s\n" % ", ".join(rctx.attr.prefixes) +
            "On Debian/Ubuntu install it with:\n" +
            "    sudo apt-get install -y libopencv-dev\n" +
            "or build from source per 'README compile OpenCV Ubuntu.txt'.",
        )

    libdir = None
    for sub in rctx.attr.libdirs:
        full = prefix + "/" + sub
        if rctx.path(full).exists and _has_glob(rctx, full + "/libopencv_core.so*"):
            libdir = full
            break
    if libdir == None:
        fail(
            "Found OpenCV headers under '%s' but no libopencv_core.so in any of: %s\n" % (
                prefix,
                ", ".join([prefix + "/" + s for s in rctx.attr.libdirs]),
            ) +
            "The OpenCV install looks incomplete. On Debian/Ubuntu:\n" +
            "    sudo apt-get install --reinstall -y libopencv-dev",
        )

    rctx.symlink(prefix + "/include", "include")
    rctx.symlink(libdir, "lib")

    present = [
        '"-l%s"' % name
        for name in _OPENCV_LINK_CANDIDATES
        if _has_glob(rctx, "%s/lib%s.so*" % (libdir, name))
    ]
    if not present:
        fail("No linkable OpenCV libraries found in '%s' (@%s)." % (libdir, rctx.name))

    rctx.file(
        "BUILD",
        _OPENCV_BUILD_TEMPLATE.format(
            prefix = prefix,
            libdir = libdir,
            linkopts = ", ".join(present),
        ),
        executable = False,
    )

local_opencv = repository_rule(
    implementation = _local_opencv_impl,
    doc = "Finds a system OpenCV 4 install (apt or source) and normalizes its layout.",
    attrs = {
        "prefixes": attr.string_list(
            default = _OPENCV_PREFIXES,
            doc = "Install prefixes to probe, in priority order.",
        ),
        "libdirs": attr.string_list(
            default = _OPENCV_LIBDIRS,
            doc = "Library subdirectories to probe under the chosen prefix.",
        ),
    },
    local = True,
    configure = True,
)
