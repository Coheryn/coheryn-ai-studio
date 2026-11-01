use coheryn_metal::{Error, ErrorKind, MemoryBudget, MemoryLease, Result};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub struct Budget(pub Arc<AtomicU64>);
struct Lease {
    used: Arc<AtomicU64>,
    bytes: u64,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl MemoryBudget for Budget {
    fn reserve(&self, bytes: u64) -> Result<Box<dyn MemoryLease>> {
        self.0
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|v| *v <= 1 << 20)
            })
            .map_err(|_| Error::new(ErrorKind::ResourceExhausted, "test GPU budget exhausted"))?;
        Ok(Box::new(Lease {
            used: Arc::clone(&self.0),
            bytes,
        }))
    }
}
