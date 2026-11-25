//! Unicode embedding of an explicitly supplied, embeddable TrueType face. Font files are
//! document inputs or the optional craft-fonts build input, never system-interface fallbacks.

use std::collections::HashMap;

use pdfcraft_cos::{Dict, Document, ObjRef, Object};
use skrifa::{
    FontRef, MetadataProvider,
    instance::{LocationRef, Size},
    raw::TableProvider,
};

use craft_pdf::fonts::wire::{
    CidMapping, FontDescriptorSpec, build_cid_encoding_cmap, build_cid_font_dict, build_font_descriptor, build_to_unicode_cmap,
    build_type0_font_dict, font_program_stream,
};

/// Compare PDF subset/PostScript names with the full face's family name.
pub fn face_key(name: &str) -> String {
    name.rsplit('+').next().unwrap_or(name).chars().filter(char::is_ascii_alphanumeric).flat_map(char::to_lowercase).collect()
}

/// Full, approved faces of the source family. This never silently selects another family.
pub fn matching_faces(name: &str) -> Vec<&'static crate::CraftFont> {
    let key = face_key(name);
    crate::CRAFT_FONTS.iter().filter(|f| face_key(f.family) == key || face_key(&f.name()) == key).collect()
}

/// Whether a complete font program can show every non-newline character.
pub fn covers(bytes: &[u8], text: &str) -> bool {
    FontRef::new(bytes).is_ok_and(|f| text.chars().all(|c| matches!(c, '\n' | '\r') || f.charmap().map(c).is_some_and(|g| g.to_u32() != 0)))
}

/// Names come from the program, never a caller's guessed family.
pub fn names(bytes: &[u8]) -> Result<(String, String), String> {
    use skrifa::string::StringId;
    validate(bytes)?;
    let f = FontRef::new(bytes).map_err(|e| e.to_string())?;
    let name = |id| {
        f.localized_strings(id)
            .find(|s| s.language() == Some("en"))
            .or_else(|| f.localized_strings(id).next())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty() && s.len() <= 128)
    };
    let family = name(StringId::FAMILY_NAME).ok_or("the font has no family name")?;
    let postscript = name(StringId::POSTSCRIPT_NAME).unwrap_or_else(|| family.replace(' ', ""));
    Ok((family, postscript))
}

/// Validate a selected external document font: require a standalone, embeddable
/// TrueType/OpenType face with a valid family name, without checking its licence metadata.
pub fn validate_document_input(bytes: &[u8]) -> Result<(), String> {
    names(bytes).map(|_| ())
}

/// Validate before allocating any PDF objects, so an invalid font cannot partly mutate a doc.
pub fn validate(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > 16 << 20 {
        return Err("the selected font exceeds 16 MiB".into());
    }
    let font = FontRef::new(bytes).map_err(|_| "select a standalone TrueType/OpenType font (not a font collection)")?;
    if font.fvar().is_ok() {
        return Err("select a static instance of this variable font before embedding".into());
    }
    if (font.glyf().is_err() && font.cff().is_err()) || font.head().is_err() || font.hhea().is_err() || font.hmtx().is_err() {
        return Err("the selected font has no usable TrueType/CFF outlines and metrics".into());
    }
    if font.os2().is_ok_and(|t| t.fs_type() & (0x0002 | 0x0004 | 0x0200) != 0) {
        return Err("this font's embedding permissions prohibit editable outline embedding".into());
    }
    let m = font.metrics(Size::unscaled(), LocationRef::default());
    if m.units_per_em == 0 || font.charmap().mappings().next().is_none() {
        return Err("the selected font has no Unicode mapping or valid units per em".into());
    }
    if font.charmap().mappings().take(65_535).count() >= 65_535 {
        return Err("the selected font has too many Unicode mappings for one PDF font".into());
    }
    if font.charmap().mappings().any(|(_, g)| g.to_u32() >= u32::from(m.glyph_count)) {
        return Err("the selected font maps Unicode to a glyph outside its program".into());
    }
    Ok(())
}

/// Embed the complete font once and provide separate CIDs for Unicode aliases. /ToUnicode,
/// /CIDToGIDMap and widths agree, including supplementary characters, so reopening needs no
/// installed font. Keeping the full program also lets subsequent edits add other glyphs.
pub fn embed(doc: &mut Document, fonts: &mut Dict, bytes: &[u8], family: &str) -> Result<String, String> {
    validate(bytes)?;
    let font = FontRef::new(bytes).map_err(|e| e.to_string())?;
    let mappings: Vec<_> = font
        .charmap()
        .mappings()
        .filter(|(_, gid)| gid.to_u32() > 0)
        .filter_map(|(cp, gid)| char::from_u32(cp).map(|c| (c.to_string(), gid)))
        .collect();
    embed_mapping(doc, fonts, bytes, family, &mappings)
}

/// Explicit glyph mapping for shaped text. A CID can represent a contextual glyph or a
/// ligature, while its ToUnicode retains the logical cluster. The full source program stays exact.
pub fn embed_mapping(
    doc: &mut Document,
    fonts: &mut Dict,
    bytes: &[u8],
    family: &str,
    mappings: &[(String, skrifa::GlyphId)],
) -> Result<String, String> {
    validate(bytes)?;
    if mappings.len() > 65_534 || mappings.iter().any(|(s, _)| s.len() > 4096) {
        return Err("shaped font mapping exceeds its budget".into());
    }
    let font = FontRef::new(bytes).map_err(|e| e.to_string())?;
    let cff = font.glyf().is_err();
    let loc = LocationRef::default();
    let m = font.metrics(Size::unscaled(), loc);
    let scale = 1000.0 / f64::from(m.units_per_em);
    let advances = font.glyph_metrics(Size::unscaled(), loc);
    let mut cff_widths = std::collections::BTreeMap::new();
    let mut gids = vec![0, 0];
    let mut widths = Vec::with_capacity(mappings.len());
    let mut cid_mappings = Vec::with_capacity(mappings.len());
    for (index, (unicode, gid)) in mappings.iter().enumerate() {
        let cid = u16::try_from(index.saturating_add(1)).map_err(|_| "font mapping limit")?;
        let glyph = u16::try_from(gid.to_u32()).map_err(|_| "font glyph limit")?;
        if glyph == 0 || u32::from(glyph) >= u32::from(m.glyph_count) {
            return Err("font contains an unavailable shaped glyph".into());
        }
        let width = advances.advance_width(*gid).unwrap_or(0.0) as f64 * scale;
        if !width.is_finite() || !(0.0..=16_000.0).contains(&width) {
            return Err("font contains an invalid advance".into());
        }
        gids.extend_from_slice(&glyph.to_be_bytes());
        widths.push(Object::Real(width));
        cff_widths.insert(glyph, width);
        cid_mappings.push(CidMapping { cid, glyph, width, unicode });
    }
    let cid_encoding = build_cid_encoding_cmap(&cid_mappings);
    let cmap = build_to_unicode_cmap(&cid_mappings);
    let name = family.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')).collect::<String>();
    if name.is_empty() || name.len() > 128 {
        return Err("the font family name is invalid".into());
    }
    let glyph_map = if cff { cid_encoding.as_bytes() } else { gids.as_slice() };
    let mut programs = ProgramMatch::new(bytes);
    if let Some(resource) = same_resource(doc, fonts, &name, &mut programs, glyph_map, cmap.as_bytes()) {
        return Ok(resource);
    }
    let bounds = m.bounds.ok_or("font has no outline bounds")?;
    // Reuse the program another resource of this page already has, else one registered on another
    // page; source subset programs stay untouched.
    let program = existing_program(doc, fonts, &mut programs).unwrap_or_else(|| font_program_stream(doc, bytes, cff));
    let descriptor = build_font_descriptor(&FontDescriptorSpec {
        font_name: &name,
        family,
        italic_angle: f64::from(m.italic_angle),
        ascent: f64::from(m.ascent) * scale,
        descent: f64::from(m.descent) * scale,
        cap_height: f64::from(m.cap_height.unwrap_or(m.ascent)) * scale,
        bbox: [bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max].map(|v| f64::from(v) * scale),
        cff,
        program,
    });
    let desc = Object::Ref(doc.add(Object::Dict(descriptor)));
    let cid = build_cid_font_dict(doc, &name, desc, cff, cff_widths, gids, widths);
    let descendant = Object::Ref(doc.add(Object::Dict(cid)));
    let top = build_type0_font_dict(doc, &name, descendant, cff.then(|| cid_encoding.into_bytes()), cmap.into_bytes());
    let mut index = 0usize;
    while fonts.contains(format!("PCFull{index}").as_bytes()) {
        index = index.saturating_add(1);
    }
    let resource = format!("PCFull{index}");
    fonts.set(resource.as_bytes().to_vec(), Object::Ref(doc.add(Object::Dict(top))));
    Ok(resource)
}

/// Whether the stream `object` refers to decodes to `expected`.
fn stream_is(doc: &Document, object: Option<&Object>, expected: &[u8]) -> bool {
    object.is_some_and(|o| match doc.resolve(o).as_ref() {
        Object::Stream(s) => s.decoded_within(expected.len()).is_ok_and(|d| d == expected),
        _ => false,
    })
}

/// Full programs one embedding decodes at most while looking for its own. A real match passes
/// the cheap checks and decodes once; many candidates that pass them come from a hostile file.
const MAX_PROGRAM_DECODES: usize = 16;

/// Recognises the full program written for `bytes`: each candidate stream decodes at most once
/// and within `bytes.len()`, and at most [`MAX_PROGRAM_DECODES`] decode in all.
struct ProgramMatch<'a> {
    bytes: &'a [u8],
    known: HashMap<ObjRef, bool>,
    decodes: usize,
}

impl<'a> ProgramMatch<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, known: HashMap::new(), decodes: 0 }
    }

    /// Whether `r`, holding `object`, is the program.
    fn is(&mut self, doc: &Document, r: ObjRef, object: &Object) -> bool {
        if let Some(&known) = self.known.get(&r) {
            return known;
        }
        let found = match object {
            Object::Stream(s)
                if s.dict.get(b"PCFullFont").is_some()
                    && s.dict.get(b"Length1").and_then(|l| doc.resolve(l).as_int()) == i64::try_from(self.bytes.len()).ok()
                    && self.decodes < MAX_PROGRAM_DECODES =>
            {
                self.decodes = self.decodes.saturating_add(1);
                s.decoded_within(self.bytes.len()).is_ok_and(|data| data == self.bytes)
            }
            _ => false,
        };
        self.known.insert(r, found);
        found
    }

    /// Whether the resource's program is this one.
    fn written(&mut self, doc: &Document, w: &Written) -> bool {
        w.program.as_ref().is_some_and(|r| self.is(doc, r, &doc.get(r)))
    }
}

/// The program for `programs.bytes` that this page's `PCFull` resources, or else the first
/// objects of the document, already have.
fn existing_program(doc: &Document, fonts: &Dict, programs: &mut ProgramMatch) -> Option<Object> {
    written(doc, fonts)
        .find(|w| programs.written(doc, w))
        .map(|w| w.program)
        .or_else(|| doc.scan_objects().take(8192).find_map(|(r, o)| programs.is(doc, r, &o).then_some(Object::Ref(r))))
}

/// A `PCFull` resource that [`embed_mapping`] wrote.
struct Written {
    resource: String,
    top: Dict,
    cid: Dict,
    /// The reference to its full program.
    program: Object,
}

/// The `PCFull` resources in `fonts`, in order.
fn written<'a>(doc: &'a Document, fonts: &'a Dict) -> impl Iterator<Item = Written> + 'a {
    fonts.iter().filter(|(key, _)| key.starts_with(b"PCFull")).take(256).filter_map(move |(key, value)| {
        let top = doc.dict(value)?;
        let cid = doc.dict(doc.resolve(top.get(b"DescendantFonts")?).as_array()?.first()?)?;
        let descriptor = doc.dict(cid.get(b"FontDescriptor")?)?;
        let program = descriptor.get(b"FontFile2").or_else(|| descriptor.get(b"FontFile3")).filter(|p| matches!(p, Object::Ref(_)))?.clone();
        Some(Written { resource: String::from_utf8_lossy(key).into_owned(), top, cid, program })
    })
}

/// A resource in `fonts` that already is this embedding: the same font name, program, glyph map
/// and Unicode map. Editing the same text again reuses it instead of adding another copy.
fn same_resource(doc: &Document, fonts: &Dict, name: &str, programs: &mut ProgramMatch, glyph_map: &[u8], unicode: &[u8]) -> Option<String> {
    written(doc, fonts)
        .find(|w| {
            let glyphs = w.cid.get(b"CIDToGIDMap").or_else(|| w.top.get(b"Encoding"));
            w.top.name(b"BaseFont") == Some(name.as_bytes())
                && stream_is(doc, w.top.get(b"ToUnicode"), unicode)
                && stream_is(doc, glyphs, glyph_map)
                && programs.written(doc, w)
        })
        .map(|w| w.resource)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdf::Metrics;
    use pdfcraft_cos::Stream;

    #[test]
    fn full_unicode_embedding_keeps_outlines_advances_and_roundtrip_codes() {
        let bytes = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        let mut doc = Document::new_empty();
        let mut fonts = Dict::new();
        let name = embed(&mut doc, &mut fonts, bytes, "Inter").unwrap();
        let font = doc.dict(fonts.get(name.as_bytes()).unwrap()).unwrap();
        let m = Metrics::from_dict(&doc, &font);
        let text = "Aé ΩЖ €";
        let codes = m.encode(text).unwrap();
        assert_eq!(m.decode(&codes), text);
        assert_eq!(crate::pdf::embedded_sfnt(&doc, &font).unwrap(), bytes);
        let sfnt = FontRef::new(bytes).unwrap();
        for (ch, (code, _)) in text.chars().zip(m.codes(&codes)) {
            let gid = sfnt.charmap().map(ch).unwrap();
            let w = sfnt.glyph_metrics(Size::unscaled(), LocationRef::default()).advance_width(gid).unwrap() as f64
                / f64::from(sfnt.metrics(Size::unscaled(), LocationRef::default()).units_per_em);
            assert!((m.width(code) - w).abs() < 1e-8);
        }
        let before = doc.object_numbers().len();
        assert!(embed(&mut doc, &mut fonts, b"broken font", "Broken").is_err());
        assert!(embed_mapping(&mut doc, &mut fonts, bytes, "Inter", &[("X".into(), skrifa::GlyphId::new(65_535))]).is_err());
        assert_eq!(doc.object_numbers().len(), before);
        let programs = || doc.scan_objects().filter(|(_, o)| matches!(o.as_ref(), Object::Stream(s) if s.dict.get(b"PCFullFont").is_some())).count();
        assert_eq!(programs(), 1);
        let mut second = Dict::new();
        embed(&mut doc, &mut second, bytes, "Inter").unwrap();
        assert_eq!(doc.scan_objects().filter(|(_, o)| matches!(o.as_ref(), Object::Stream(s) if s.dict.get(b"PCFullFont").is_some())).count(), 1);
        assert_eq!(face_key("ABCDEF+Droid-Sans-Fallback"), face_key("Droid Sans Fallback"));
    }

    fn replace_name_table(sfnt: &[u8], records: &[(u16, &str)]) -> Vec<u8> {
        let mut sorted = records.to_vec();
        sorted.sort_by_key(|&(id, _)| id);
        let mut name = Vec::new();
        name.extend_from_slice(&0u16.to_be_bytes());
        name.extend_from_slice(&(sorted.len() as u16).to_be_bytes());
        name.extend_from_slice(&((6 + 12 * sorted.len()) as u16).to_be_bytes());
        let mut strings = Vec::new();
        for (id, text) in sorted {
            let encoded: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
            for field in [3u16, 1, 0x0409, id, encoded.len() as u16, strings.len() as u16] {
                name.extend_from_slice(&field.to_be_bytes());
            }
            strings.extend_from_slice(&encoded);
        }
        name.extend_from_slice(&strings);

        let num_tables = u16::from_be_bytes([sfnt[4], sfnt[5]]) as usize;
        let mut tables: Vec<([u8; 4], Vec<u8>)> = (0..num_tables)
            .filter_map(|i| {
                let rec = 12 + i * 16;
                let tag: [u8; 4] = sfnt[rec..rec + 4].try_into().unwrap();
                if &tag == b"name" {
                    return None;
                }
                let offset = u32::from_be_bytes(sfnt[rec + 8..rec + 12].try_into().unwrap()) as usize;
                let len = u32::from_be_bytes(sfnt[rec + 12..rec + 16].try_into().unwrap()) as usize;
                Some((tag, sfnt[offset..offset + len].to_vec()))
            })
            .collect();
        tables.push((*b"name", name));
        tables.sort_by_key(|(tag, _)| *tag);

        let count = tables.len() as u16;
        let selector = count.ilog2() as u16;
        let search_range = (1u16 << selector) * 16;
        let mut out = Vec::new();
        out.extend_from_slice(&sfnt[0..4]);
        out.extend_from_slice(&count.to_be_bytes());
        out.extend_from_slice(&search_range.to_be_bytes());
        out.extend_from_slice(&selector.to_be_bytes());
        out.extend_from_slice(&(count * 16 - search_range).to_be_bytes());

        let mut offset = 12 + 16 * tables.len();
        let mut payload = Vec::new();
        for (tag, bytes) in &tables {
            let mut padded = bytes.clone();
            padded.resize(padded.len().next_multiple_of(4), 0);
            let sum = padded.as_chunks::<4>().0.iter().fold(0u32, |s, b| s.wrapping_add(u32::from_be_bytes(*b)));
            out.extend_from_slice(tag);
            out.extend_from_slice(&sum.to_be_bytes());
            out.extend_from_slice(&(offset as u32).to_be_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            offset += padded.len();
            payload.extend_from_slice(&padded);
        }
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn document_input_accepts_any_valid_font_without_licence_checks() {
        let bytes = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        assert!(validate_document_input(bytes).is_ok(), "an OFL font passes");

        let unlicensed = replace_name_table(bytes, &[(1, "Unlicensed Sans"), (2, "Regular"), (6, "UnlicensedSans-Regular")]);
        assert!(validate_document_input(&unlicensed).is_ok(), "a font without LICENSE_DESCRIPTION passes");

        let proprietary_adobe = replace_name_table(
            bytes,
            &[
                (0, "Copyright 2026 Adobe"),
                (1, "Proprietary Sans"),
                (2, "Regular"),
                (6, "ProprietarySans-Regular"),
                (8, "Adobe"),
                (13, "Proprietary Commercial License"),
            ],
        );
        assert!(validate_document_input(&proprietary_adobe).is_ok(), "Adobe and proprietary fonts pass");

        let noto_cjk = replace_name_table(bytes, &[(1, "Noto Sans CJK JP"), (2, "Regular"), (6, "NotoSansCJKjp-Regular")]);
        assert!(validate_document_input(&noto_cjk).is_ok(), "NotoSansCJK without license passes");

        let source_han = replace_name_table(bytes, &[(1, "Source Han Sans"), (2, "Regular"), (6, "SourceHanSans-Regular")]);
        assert!(validate_document_input(&source_han).is_ok(), "SourceHanSans without license passes");

        let mut doc = Document::new_empty();
        let mut fonts = Dict::new();
        assert!(embed(&mut doc, &mut fonts, &unlicensed, "UnlicensedSans-Regular").is_ok());

        let no_family = replace_name_table(bytes, &[]);
        assert_eq!(validate_document_input(&no_family).unwrap_err(), "the font has no family name");
        let err = validate_document_input(b"not a font").unwrap_err();
        assert!(!err.contains("openly licensed") && !err.contains("licence"), "{err}");
    }

    #[test]
    fn embedding_the_same_text_again_reuses_its_resource_and_program() {
        let bytes = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        let face = FontRef::new(bytes).unwrap();
        let glyph = |c| face.charmap().map(c).unwrap();
        let programs = |doc: &Document| {
            doc.scan_objects().filter(|(_, o)| matches!(o.as_ref(), Object::Stream(s) if s.dict.get(b"PCFullFont").is_some())).count()
        };
        // Objects before the program push it past the first 8192 that the document-wide search reads.
        let mut doc = Document::new_empty();
        for _ in 0..9000 {
            doc.add(Object::Null);
        }
        let mut fonts = Dict::new();
        let full = embed(&mut doc, &mut fonts, bytes, "Inter").unwrap();
        let objects = doc.scan_objects().count();
        assert_eq!(embed(&mut doc, &mut fonts, bytes, "Inter").unwrap(), full, "the same embedding is the same resource");
        let shaped = [("fi".to_string(), glyph('f')), ("é".to_string(), glyph('é'))];
        let first = embed_mapping(&mut doc, &mut fonts, bytes, "Inter", &shaped).unwrap();
        assert_ne!(first, full, "another mapping is another resource");
        assert_eq!(embed_mapping(&mut doc, &mut fonts, bytes, "Inter", &shaped).unwrap(), first);
        assert_eq!(embed_mapping(&mut doc, &mut fonts, bytes, "Inter", &shaped[..1]).map(|r| r != first), Ok(true), "a different mapping");
        assert_eq!(programs(&doc), 1, "one program for every mapping");
        assert_eq!(fonts.iter().count(), 3);
        assert!(doc.scan_objects().count() > objects);
        // Another writer may have compressed the streams since; the page still recognises them.
        for (r, o) in doc.scan_objects().collect::<Vec<_>>() {
            if let Object::Stream(st) = o.as_ref()
                && st.dict.get(b"Filter").is_none()
            {
                let data = st.decoded().unwrap();
                doc.set(r, Object::Stream(Stream::flate(st.dict.clone(), &data)));
            }
        }
        let objects = doc.scan_objects().count();
        assert_eq!(embed(&mut doc, &mut fonts, bytes, "Inter").unwrap(), full);
        assert_eq!(embed_mapping(&mut doc, &mut fonts, bytes, "Inter", &shaped).unwrap(), first);
        assert_eq!(doc.scan_objects().count(), objects, "nothing new is written");
        // A different program under the same name is not taken for this one.
        let other = include_bytes!("../../../assets/fonts/Inter-Medium.ttf");
        let mut more = fonts.clone();
        let medium = embed(&mut doc, &mut more, other, "Inter").unwrap();
        assert!(!fonts.contains(medium.as_bytes()), "{medium} is new");
        assert_eq!(programs(&doc), 2);
    }

    #[test]
    fn the_program_search_decodes_each_candidate_once_and_few_in_all() {
        let bytes = b"the program".as_slice();
        let decoy = |doc: &mut Document| {
            let mut d = Dict::new();
            d.set(b"PCFullFont".to_vec(), Object::Bool(true));
            d.set(b"Length1".to_vec(), Object::Int(bytes.len() as i64));
            doc.add(Object::Stream(Stream::flate(d, &[0; 4096])))
        };
        let resource = |doc: &mut Document, program: ObjRef| {
            let mut descriptor = Dict::new();
            descriptor.set(b"FontFile2".to_vec(), Object::Ref(program));
            let mut cid = Dict::new();
            cid.set(b"FontDescriptor".to_vec(), Object::Dict(descriptor));
            let mut top = Dict::new();
            top.set(b"DescendantFonts".to_vec(), Object::Array(vec![Object::Dict(cid)]));
            doc.add(Object::Dict(top))
        };
        // 200 resources sharing one decoy program decode it once.
        let mut doc = Document::new_empty();
        let shared = decoy(&mut doc);
        let mut fonts = Dict::new();
        for i in 0..200 {
            let r = resource(&mut doc, shared);
            fonts.set(format!("PCFull{i}").into_bytes(), Object::Ref(r));
        }
        let mut programs = ProgramMatch::new(bytes);
        assert_eq!(existing_program(&doc, &fonts, &mut programs), None);
        assert_eq!(programs.decodes, 1);
        // 100 distinct decoys decode a bounded number of times.
        let mut doc = Document::new_empty();
        let mut fonts = Dict::new();
        for i in 0..100 {
            let program = decoy(&mut doc);
            let r = resource(&mut doc, program);
            fonts.set(format!("PCFull{i}").into_bytes(), Object::Ref(r));
        }
        let mut programs = ProgramMatch::new(bytes);
        assert_eq!(existing_program(&doc, &fonts, &mut programs), None);
        assert_eq!(programs.decodes, MAX_PROGRAM_DECODES);
    }

    #[test]
    fn the_full_program_is_written_compressed_and_another_page_reuses_it() {
        let bytes = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        let mut doc = Document::new_empty();
        let mut first = Dict::new();
        embed(&mut doc, &mut first, bytes, "Inter").unwrap();
        let program = || {
            doc.scan_objects()
                .find_map(|(_, o)| match o.as_ref() {
                    Object::Stream(s) if s.dict.get(b"PCFullFont").is_some() => Some(s.clone()),
                    _ => None,
                })
                .unwrap()
        };
        let stream = program();
        assert_eq!(stream.dict.name(b"Filter"), Some(b"FlateDecode".as_slice()));
        assert!(stream.raw.len() < bytes.len(), "{} of {} bytes", stream.raw.len(), bytes.len());
        assert_eq!(stream.decoded().unwrap(), bytes);
        // Another page's resources don't have it, so it is found among the document's objects.
        let objects = doc.scan_objects().count();
        let mut second = Dict::new();
        embed(&mut doc, &mut second, bytes, "Inter").unwrap();
        let programs = doc.scan_objects().filter(|(_, o)| matches!(o.as_ref(), Object::Stream(s) if s.dict.get(b"PCFullFont").is_some())).count();
        assert_eq!(programs, 1, "one program for both pages");
        assert!(doc.scan_objects().count() > objects, "the second page gets its own resource");
    }

    #[test]
    fn custom_cmap_wire_builders_round_trip_through_metrics_across_chunks_and_surrogates() {
        let mut doc = Document::new_empty();
        let mut cff_widths = std::collections::BTreeMap::new();
        let mut entries: Vec<(u16, u16, f64, String)> =
            vec![(1, 301, 512.0, "A".into()), (2, 302, 625.0, "é".into()), (3, 303, 750.0, "fi".into()), (4, 304, 880.0, "𠀋".into())];
        // Exceed the 100-entry CMap chunk size so multi-block begincidchar/beginbfchar output is exercised.
        for i in 5..=125u16 {
            let ch = char::from_u32(0x4E00 + u32::from(i)).unwrap();
            entries.push((i, 300 + i, 500.0 + f64::from(i), ch.to_string()));
        }
        for &(_, glyph, width, _) in &entries {
            cff_widths.insert(glyph, width);
        }
        let cid_mappings: Vec<_> = entries
            .iter()
            .map(|(cid, glyph, width, unicode)| CidMapping { cid: *cid, glyph: *glyph, width: *width, unicode: unicode.as_str() })
            .collect();
        let cid_encoding = build_cid_encoding_cmap(&cid_mappings);
        let to_unicode = build_to_unicode_cmap(&cid_mappings);
        let program = font_program_stream(&mut doc, b"OTTO", true);
        let descriptor = build_font_descriptor(&FontDescriptorSpec {
            font_name: "CustomCffTest",
            family: "CustomCffTest",
            italic_angle: 0.0,
            ascent: 800.0,
            descent: -200.0,
            cap_height: 700.0,
            bbox: [0.0, -200.0, 1000.0, 800.0],
            cff: true,
            program,
        });
        let desc = Object::Ref(doc.add(Object::Dict(descriptor)));
        let cid = build_cid_font_dict(&mut doc, "CustomCffTest", desc, true, cff_widths, Vec::new(), Vec::new());
        let descendant = Object::Ref(doc.add(Object::Dict(cid)));
        let top = build_type0_font_dict(&mut doc, "CustomCffTest", descendant, Some(cid_encoding.into_bytes()), to_unicode.into_bytes());

        let m = Metrics::from_dict(&doc, &top);
        for &(cid_code, glyph, width, ref expected) in &entries {
            let code = u32::from(cid_code);
            let bytes = cid_code.to_be_bytes();
            assert_eq!(m.decode(&bytes), *expected, "cid {cid_code} ToUnicode");
            assert_eq!(m.cid(code), u32::from(glyph), "cid {cid_code} PCGlyphs mapping");
            assert!((m.width(code) - width / 1000.0).abs() < 1e-8, "cid {cid_code} width");
        }
    }
}
