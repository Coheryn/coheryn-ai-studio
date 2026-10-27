fn main() {
    let compiler =
        std::process::Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
            .arg("--version")
            .output()
            .expect("rustc version");
    println!(
        "cargo:rustc-env=COHERYN_RUSTC={}",
        String::from_utf8_lossy(&compiler.stdout).trim()
    );
}
