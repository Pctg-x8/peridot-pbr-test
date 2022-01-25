const GRID_SIZE: usize = 10;
const GRID_AXIS_LENGTH: f32 = 100.0;
const fn colored_line_mesh(
    from: peridot::math::Vector4F32,
    to: peridot::math::Vector4F32,
    color: peridot::math::Vector4F32,
) -> [peridot::ColoredVertex; 2] {
    [
        peridot::ColoredVertex { pos: from, color },
        peridot::ColoredVertex { pos: to, color },
    ]
}
pub const GRID_MESH_LINE_COUNT: usize = (GRID_SIZE * 2) * (GRID_SIZE * 2) + 2 + 3;
pub fn build_grid_mesh_into(dest: &mut [[peridot::ColoredVertex; 2]]) {
    use peridot::math::Vector4;

    let mut ptr = 0;

    for x in 1..=GRID_SIZE {
        dest[ptr] = colored_line_mesh(
            Vector4(x as f32, 0.0, -(GRID_SIZE as f32), 1.0),
            Vector4(x as f32, 0.0, GRID_SIZE as f32, 1.0),
            Vector4(0.375, 0.375, 0.375, 1.0),
        );
        ptr += 1;

        dest[ptr] = colored_line_mesh(
            Vector4(-(x as f32), 0.0, -(GRID_SIZE as f32), 1.0),
            Vector4(-(x as f32), 0.0, GRID_SIZE as f32, 1.0),
            Vector4(0.375, 0.375, 0.375, 1.0),
        );
        ptr += 1;

        dest[ptr] = colored_line_mesh(
            Vector4(-(GRID_SIZE as f32), 0.0, -(x as f32), 1.0),
            Vector4(GRID_SIZE as f32, 0.0, -(x as f32), 1.0),
            Vector4(0.375, 0.375, 0.375, 1.0),
        );
        ptr += 1;

        dest[ptr] = colored_line_mesh(
            Vector4(-(GRID_SIZE as f32), 0.0, x as f32, 1.0),
            Vector4(GRID_SIZE as f32, 0.0, x as f32, 1.0),
            Vector4(0.375, 0.375, 0.375, 1.0),
        );
        ptr += 1;
    }

    dest[ptr] = colored_line_mesh(
        Vector4(0.0, 0.0, -(GRID_SIZE as f32), 1.0),
        Vector4(0.0, 0.0, GRID_SIZE as f32, 1.0),
        Vector4(0.375, 0.375, 0.375, 1.0),
    );
    ptr += 1;
    dest[ptr] = colored_line_mesh(
        Vector4(-(GRID_SIZE as f32), 0.0, 0.0, 1.0),
        Vector4(GRID_SIZE as f32, 0.0, 0.0, 1.0),
        Vector4(0.375, 0.375, 0.375, 1.0),
    );
    ptr += 1;

    dest[ptr] = colored_line_mesh(
        Vector4(0.0, 0.0, 0.0, 1.0),
        Vector4(GRID_AXIS_LENGTH as f32, 0.0, 0.0, 1.0),
        Vector4(1.0, 0.0, 0.0, 1.0),
    );
    ptr += 1;
    dest[ptr] = colored_line_mesh(
        Vector4(0.0, 0.0, 0.0, 1.0),
        Vector4(0.0, GRID_AXIS_LENGTH as f32, 0.0, 1.0),
        Vector4(0.0, 1.0, 0.0, 1.0),
    );
    ptr += 1;
    dest[ptr] = colored_line_mesh(
        Vector4(0.0, 0.0, 0.0, 1.0),
        Vector4(0.0, 0.0, GRID_AXIS_LENGTH as f32, 1.0),
        Vector4(0.0, 0.0, 1.0, 1.0),
    );
}

fn normalize3(v: peridot::math::Vector4<f32>) -> peridot::math::Vector4F32 {
    let peridot::math::Vector3(x, y, z) = peridot::math::Vector3::from(v).normalize();
    peridot::math::Vector4(x, y, z, 1.0)
}

#[repr(C)]
#[derive(Clone)]
pub struct VertexWithNormals {
    pos: peridot::math::Vector4F32,
    normal: peridot::math::Vector4F32,
    tangent: peridot::math::Vector4F32,
    binormal: peridot::math::Vector4F32,
}
pub struct UnitIcosphere {
    pub vertices: Vec<VertexWithNormals>,
    pub indices: Vec<u16>,
}
impl UnitIcosphere {
    pub fn base() -> Self {
        // recip of golden number(short side)
        let ph = 2.0f32 / (1.0f32 + 5.0f32.sqrt());

        let xz_planes = vec![
            peridot::math::Vector4(-ph, 0.0, -1.0, 1.0),
            peridot::math::Vector4(ph, 0.0, -1.0, 1.0),
            peridot::math::Vector4(ph, 0.0, 1.0, 1.0),
            peridot::math::Vector4(-ph, 0.0, 1.0, 1.0),
        ];
        let xy_planes = vec![
            peridot::math::Vector4(-1.0, -ph, 0.0, 1.0),
            peridot::math::Vector4(-1.0, ph, 0.0, 1.0),
            peridot::math::Vector4(1.0, ph, 0.0, 1.0),
            peridot::math::Vector4(1.0, -ph, 0.0, 1.0),
        ];
        let yz_planes = vec![
            peridot::math::Vector4(0.0, -1.0, -ph, 1.0),
            peridot::math::Vector4(0.0, -1.0, ph, 1.0),
            peridot::math::Vector4(0.0, 1.0, ph, 1.0),
            peridot::math::Vector4(0.0, 1.0, -ph, 1.0),
        ];

        Self {
            vertices: xz_planes
                .into_iter()
                .chain(xy_planes.into_iter())
                .chain(yz_planes.into_iter())
                .map(|p| {
                    let n3 = peridot::math::Vector3::from(p.clone()).normalize();
                    let ax = peridot::math::Vector3::up().cross(&n3);
                    let q = peridot::math::Quaternion(
                        ax.0,
                        ax.1,
                        ax.2,
                        1.0 + peridot::math::Vector3::up().dot(n3.clone()),
                    )
                    .normalize();
                    let rot = peridot::math::Matrix4::from(q);

                    VertexWithNormals {
                        normal: peridot::math::Vector4(n3.0, n3.1, n3.2, 0.0),
                        tangent: rot.clone() * peridot::math::Vector4(1.0, 0.0, 0.0, 0.0),
                        binormal: rot * peridot::math::Vector4(0.0, 0.0, 1.0, 0.0),
                        pos: normalize3(p),
                    }
                })
                .collect(),
            indices: vec![
                0, 1, 8, 0, 1, 11, 2, 3, 9, 2, 3, 10, 4, 5, 0, 4, 5, 3, 6, 7, 1, 6, 7, 2, 8, 9, 4,
                8, 9, 7, 10, 11, 5, 10, 11, 6, 8, 0, 4, 8, 1, 7, 11, 0, 5, 11, 1, 6, 9, 2, 7, 9, 3,
                4, 10, 2, 6, 10, 3, 5,
            ],
        }
    }

    pub fn subdivide(&self) -> Self {
        let mut midpoints = std::collections::HashMap::<(u16, u16), u16>::new();
        let mut vertices = self.vertices.clone();
        let mut midpoint_index = |index1: u16, index2: u16| {
            *midpoints.entry((index1, index2)).or_insert_with(|| {
                let index = vertices.len() as u16;
                let p = normalize3(
                    (vertices[index1 as usize].pos + vertices[index2 as usize].pos) * 0.5,
                );

                let n3 = peridot::math::Vector3::from(p.clone()).normalize();
                let ax = peridot::math::Vector3::up().cross(&n3);
                let q = peridot::math::Quaternion(
                    ax.0,
                    ax.1,
                    ax.2,
                    1.0 + peridot::math::Vector3::up().dot(n3.clone()),
                )
                .normalize();
                let rot = peridot::math::Matrix4::from(q);

                vertices.push(VertexWithNormals {
                    normal: peridot::math::Vector4(n3.0, n3.1, n3.2, 0.0),
                    tangent: rot.clone() * peridot::math::Vector4(1.0, 0.0, 0.0, 0.0),
                    binormal: rot * peridot::math::Vector4(0.0, 0.0, 1.0, 0.0),
                    pos: p,
                });

                index
            })
        };
        let mut indices = Vec::new();

        for tri in self.indices.chunks_exact(3) {
            let mp01 = midpoint_index(tri[0], tri[1]);
            let mp12 = midpoint_index(tri[1], tri[2]);
            let mp20 = midpoint_index(tri[2], tri[0]);

            indices.extend(vec![
                tri[0], mp20, mp01, tri[1], mp12, mp01, tri[2], mp20, mp12, mp01, mp12, mp20,
            ]);
        }

        Self { vertices, indices }
    }
}

pub const UI_FILL_RECT_COUNT: usize = 4 * 4;
pub const UI_FILL_RECT_INDEX_COUNT: usize = 6 * 4;
pub const UI_FILL_RECT_BORDER_INDEX_COUNT: usize = 2 * 4;
pub const UI_FILL_RECT_BORDER_INDICES: &'static [u16; UI_FILL_RECT_BORDER_INDEX_COUNT] =
    &[1, 3, 5, 7, 9, 11, 13, 15];
pub const UI_FILL_RECT_INDICES: &'static [u16; UI_FILL_RECT_INDEX_COUNT] = &[
    0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 14, 13, 14, 15,
];
