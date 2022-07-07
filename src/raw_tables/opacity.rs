pub(crate) struct RawOpacityTable(pub &'static [u8]);

pub(crate) const RAW_TABLES: RawOpacityTable = RawOpacityTable(include_bytes!("opacs.bindata"));
