use crate::index::Range;

pub(crate) struct AllRawTables {
    pub metallicities: Range,
    pub tables: &'static [MetalRawTables],
}

pub(crate) struct MetalRawTables {
    pub h_fracs: Range,
    pub tables: &'static [RawTable],
}

pub(crate) struct RawTable(pub &'static [u8]);

pub(crate) static RAW_TABLES: AllRawTables = AllRawTables {
    metallicities: Range::new(0.0, 0.02, 3),
    tables: &[
        MetalRawTables {
            h_fracs: Range::new(0.0, 1.0, 6),
            tables: &[
                RawTable(include_bytes!("output_DE_z0.00x0.00.bindata")),
                RawTable(include_bytes!("output_DE_z0.00x0.20.bindata")),
                RawTable(include_bytes!("output_DE_z0.00x0.40.bindata")),
                RawTable(include_bytes!("output_DE_z0.00x0.60.bindata")),
                RawTable(include_bytes!("output_DE_z0.00x0.80.bindata")),
                RawTable(include_bytes!("output_DE_z0.00x1.00.bindata")),
            ],
        },
        MetalRawTables {
            h_fracs: Range::new(0.0, 0.2, 5),
            tables: &[
                RawTable(include_bytes!("output_DE_z0.02x0.00.bindata")),
                RawTable(include_bytes!("output_DE_z0.02x0.20.bindata")),
                RawTable(include_bytes!("output_DE_z0.02x0.40.bindata")),
                RawTable(include_bytes!("output_DE_z0.02x0.60.bindata")),
                RawTable(include_bytes!("output_DE_z0.02x0.80.bindata")),
            ],
        },
        MetalRawTables {
            h_fracs: Range::new(0.0, 0.2, 5),
            tables: &[
                RawTable(include_bytes!("output_DE_z0.04x0.00.bindata")),
                RawTable(include_bytes!("output_DE_z0.04x0.20.bindata")),
                RawTable(include_bytes!("output_DE_z0.04x0.40.bindata")),
                RawTable(include_bytes!("output_DE_z0.04x0.60.bindata")),
                RawTable(include_bytes!("output_DE_z0.04x0.80.bindata")),
            ],
        },
    ],
};
