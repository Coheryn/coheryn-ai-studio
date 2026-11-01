fn main() {
    if std::env::var_os("CARGO_FEATURE_METAL").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        // Required by MTLCreateSystemDefaultDevice, including headless programs.
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
}
