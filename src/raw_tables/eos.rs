use std::io::{self, Read};

use ndarray::{s, Array3};

use crate::{fort_unfmt::read_fort_record, index::Range};

pub(crate) struct AllRawTables {
    pub metallicities: Range,
    pub tables: &'static [MetalRawTables],
}

pub(crate) struct MetalRawTables {
    pub h_fracs: Range,
    pub tables: &'static [RawTable],
}

pub(crate) struct RawTable(pub &'static [u8]);

pub(crate) const RAW_TABLES: AllRawTables = AllRawTables {
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

pub(crate) struct RawTableContent {
    pub(crate) log_volume: Range,
    pub(crate) log_energy: Range,
    pub(crate) values: Array3<f64>,
}

impl RawTableContent {
    fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut shape = [0_u32; 3]; // ne, nv, nvars
        read_fort_record(&mut reader, &mut shape)?;
        let shape = shape.map(|e| e as usize);

        let mut log_volume = vec![0.0; shape[1]];
        read_fort_record(&mut reader, &mut log_volume)?;
        let log_volume = Range::from_slice(&log_volume)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut log_energy = vec![0.0; shape[0]];
        read_fort_record(&mut reader, &mut log_energy)?;
        let log_energy = Range::from_slice(&log_energy)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut values = Array3::zeros(shape);
        for i_v in 0..shape[1] {
            for i_e in 0..shape[0] {
                let mut slc = values.slice_mut(s![i_e, i_v, ..]);
                let raw_slc = slc.as_slice_mut().expect("values should be contiguous");
                read_fort_record(&mut reader, raw_slc)?;
            }
        }

        Ok(Self {
            log_volume,
            log_energy,
            values,
        })
    }
}

impl From<&RawTable> for RawTableContent {
    fn from(rawtbl: &RawTable) -> Self {
        Self::read_from(rawtbl.0).expect("raw tables are well-formed")
    }
}
