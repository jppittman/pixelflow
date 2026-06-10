//! Simple quad mesh parser for subdivision surfaces.
//!
//! Focused on Catmull-Clark subdivision, which requires:
//! - Pure quad topology (no triangles or n-gons)
//! - Vertex positions
//! - Face connectivity (indices into vertex array)
//! - Valence computation (edges per vertex)

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A 3D point in space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3 {
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// A quad face defined by 4 vertex indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    /// Vertex indices [v0, v1, v2, v3] in CCW order
    pub vertices: [usize; 4],
}

impl Quad {
    #[must_use]
    pub fn new(v0: usize, v1: usize, v2: usize, v3: usize) -> Self {
        Self {
            vertices: [v0, v1, v2, v3],
        }
    }
}

/// A quad mesh - the control cage for subdivision surfaces.
#[derive(Clone, Debug)]
pub struct QuadMesh {
    /// Vertex positions
    pub vertices: Vec<Point3>,
    /// Face connectivity
    pub faces: Vec<Quad>,
    /// Valence (edge count) per vertex
    pub valence: Vec<usize>,
}

impl QuadMesh {
    /// Create an empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            valence: Vec::new(),
        }
    }

    /// Parse an OBJ file containing quad faces only.
    ///
    /// Returns error if the file contains non-quad faces.
    pub fn from_obj_file(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        Self::from_obj(BufReader::new(file))
    }

    /// Parse OBJ from any reader.
    pub fn from_obj<R: Read>(reader: BufReader<R>) -> Result<Self, String> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line =
                line_result.map_err(|e| format!("IO error at line {}: {}", line_num + 1, e))?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "v" => {
                    if parts.len() < 4 {
                        return Err(format!("Line {}: vertex needs 3 coordinates", line_num + 1));
                    }
                    let x = parts[1]
                        .parse()
                        .map_err(|_| format!("Line {}: invalid x coordinate", line_num + 1))?;
                    let y = parts[2]
                        .parse()
                        .map_err(|_| format!("Line {}: invalid y coordinate", line_num + 1))?;
                    let z = parts[3]
                        .parse()
                        .map_err(|_| format!("Line {}: invalid z coordinate", line_num + 1))?;
                    vertices.push(Point3::new(x, y, z));
                }
                "f" => {
                    if parts.len() != 5 {
                        return Err(format!(
                            "Line {}: only quad faces supported (got {} vertices)",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }

                    let mut indices = [0usize; 4];
                    for (i, part) in parts[1..5].iter().enumerate() {
                        // OBJ indices can be "v", "v/vt", "v/vt/vn", "v//vn"
                        // We only care about vertex index (first number)
                        let idx_str = part.split('/').next().unwrap();
                        let idx: usize = idx_str
                            .parse()
                            .map_err(|_| format!("Line {}: invalid vertex index", line_num + 1))?;

                        // OBJ uses 1-based indexing
                        if idx == 0 {
                            return Err(format!("Line {}: vertex index cannot be 0", line_num + 1));
                        }
                        indices[i] = idx - 1;
                    }

                    faces.push(Quad::new(indices[0], indices[1], indices[2], indices[3]));
                }
                _ => {
                    // Ignore other directives (vt, vn, mtllib, usemtl, etc)
                }
            }
        }

        if vertices.is_empty() {
            return Err("No vertices found in OBJ file".to_string());
        }
        if faces.is_empty() {
            return Err("No faces found in OBJ file".to_string());
        }

        let valence = compute_valence(&vertices, &faces);
        Ok(Self {
            vertices,
            faces,
            valence,
        })
    }

    /// Get vertex count.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get face count.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if vertex index is valid.
    #[must_use]
    pub fn is_valid_vertex(&self, idx: usize) -> bool {
        idx < self.vertices.len()
    }
}

impl Default for QuadMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute valence (edge count) for each vertex.
///
/// For a closed manifold quad mesh, each vertex appears in exactly
/// `valence` faces, and has `valence` edges meeting at it.
fn compute_valence(vertices: &[Point3], faces: &[Quad]) -> Vec<usize> {
    let mut valence = vec![0; vertices.len()];

    for face in faces {
        for &v in &face.vertices {
            valence[v] += 1;
        }
    }

    valence
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_simple_quad() {
        let obj = "
# Simple cube (partial)
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.faces[0].vertices, [0, 1, 2, 3]);
    }

    #[test]
    fn test_valence_computation() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.5 0.5 1.0
f 1 2 5 5
f 2 3 5 5
f 3 4 5 5
f 4 1 5 5
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        // Vertex 4 (index 4) appears in all 4 faces (degenerate quads here, but still counts)
        assert_eq!(mesh.valence[4], 8); // Each face contributes 2 references
    }

    #[test]
    fn test_reject_triangles() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
";
        let result = QuadMesh::from_obj(BufReader::new(Cursor::new(obj)));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("quad faces"));
    }

    #[test]
    fn test_texture_coordinates_ignored() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1 2/2 3/3 4/4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.faces[0].vertices, [0, 1, 2, 3]);
    }
}
