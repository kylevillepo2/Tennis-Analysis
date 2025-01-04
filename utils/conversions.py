def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixels(meters, referernce_height_in_meters, reference_pixels):
    return (meters * reference_pixels) / referernce_height_in_meters