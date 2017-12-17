def create_region_json(filename, object_name, x, y, w, h, metadata, text, region_type='A', full_frame=False):
    return {
        'target': 'filename',
        'filename': filename,
        'region_type': region_type,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'full_frame':full_frame,
        'metadata': metadata,
        'text': text,
        'object_name': object_name,
    }
