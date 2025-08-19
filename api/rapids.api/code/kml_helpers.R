# This script contains functions used for extracting place names from the KMLs

kml_to_csv <- function(kml_path) {
  doc <- xml2::read_xml(kml_path)
  
  ns <- xml2::xml_ns(doc)
  
  # Find all Placemark nodes using the XML namespace
  placemarks <- xml2::xml_find_all(doc, ".//d1:Placemark", ns)
  
  # Iterate over placemarks
  all_points <- lapply(placemarks, function(pm) {
    name_node <- xml2::xml_find_first(pm, "./d1:name", ns)
    name <- xml2::xml_text(name_node)
    
    # Get the polygons within each placemark
    coords_nodes <- xml2::xml_find_all(pm, ".//d1:coordinates", ns)
    
    # Get a representative coordinate as the center of each image for each
    # rapid feature reported as a polygon
    lapply(coords_nodes, function(cn) {
      coords_mat <- parse_coords(xml2::xml_text(cn))
      avg_lon <- mean(coords_mat[,1])
      avg_lat <- mean(coords_mat[,2])
      tibble::tibble(name = name, longitude = avg_lon, latitude = avg_lat)
    })
  })
  
  # Combine all points into a single data.frame
  pts_df <- dplyr::bind_rows(unlist(all_points, recursive = FALSE))
  # Round coordinates to 7 decimal places; any further digits are below the
  # pixel-level resolution of the Maps API
  pts_df <- dplyr::mutate(pts_df,
                          longitude = round(longitude, digits = 7),
                          latitude = round(latitude, digits = 7))
  pts_df
}

# Helper function to convert a character string of coordinate pairs to a matrix 
# of numeric coordinate pairs
parse_coords <- function(coord_text) {
  points <- stringr::str_split(trimws(coord_text), " ")[[1]]
  coords <- t(sapply(points, function(p) {
    nums <- as.numeric(stringr::str_split(p, ",")[[1]])
    c(nums[1], nums[2])
  }))
  coords
}

# Wrapper around xml2::xml_find_first that returns NA if the given
# property is empty. Used to ensure that the indexes of the place-name vector 
# and coordinate list for the OSM rapids align
get_name <- function(placemark, ns) {
  node <- xml2::xml_find_first(placemark, "d1:name", ns)
  if (length(node) == 0) {
    return(NA_character_)
  }
  xml2::xml_text(node)
}