#![no_main]
use coheryn_engine::{Cancellation,safetensors::{self,Limits}};
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data:&[u8]|{
    if data.len()>64<<10{return;}
    let limits=Limits{header_bytes:32<<10,index_bytes:32<<10,aggregate_header_bytes:64<<10,tensors:256,shards:4,rank:8};
    if let Ok(info)=safetensors::inspect_reader(std::io::Cursor::new(data),data.len() as u64,limits,&Cancellation::default()){
        assert!(info.header_bytes<=limits.header_bytes);assert!(info.tensors.len()<=limits.tensors);
        for t in info.tensors{assert!(t.shape.len()<=limits.rank);assert!(t.offset.checked_add(t.bytes).is_some_and(|end|end<=data.len() as u64));}
    }
});
