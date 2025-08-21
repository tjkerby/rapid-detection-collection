# Script for extracting desired image locations from KML files for the
# National Hydrography Dataset (NHD) and OpenStreetMap (OSM) known rapids
# locations databases using the xml2 package

# Before running this script, download the following files from the CIRRUS data
# release and place them in the project root directory 
# - NHDArea_rapids_slope.kml
# - OSMrapidsAll.kml

source("code/kml_helpers.R")
#-------------------------------------------------------------------------------
# NHD
# Use custom function to get a single representative coordinate from each
# rapids feature
nhd_df <- kml_to_csv("NHDArea_rapids_slope.kml") |>
  dplyr::mutate(name = paste0("nhd", name))

#-------------------------------------------------------------------------------
# OSM
# Read OSM KML file
doc_osm <- xml2::read_xml("OSMrapidsAll.kml")

ns_osm <- xml2::xml_ns(doc_osm)

# Find all Placemark nodes using the namespace
placemarks_osm <- xml2::xml_find_all(doc_osm, ".//d1:Placemark", ns_osm)

# Extract coordinates
osm_coords <- xml2::xml_text(
  xml2::xml_find_first(placemarks_osm, ".//d1:coordinates", ns_osm)
)

# Use a wrapper around xml2::xml_find_first that returns NA if the given
# property is empty
osm_names <- sapply(placemarks_osm, get_name, ns = ns_osm)

# Split the coordinate pairs within each placemark so that we have a list of
# character vectors with each element being a coordinate pair
osm_coords_lst <- stringr::str_split(osm_coords, " ")

osm_df <- tibble::tibble(name = osm_names, coords = osm_coords_lst)

# Give each coordinate pair its own row while keeping the place name (if any) 
# associated with the location
osm_df <- osm_df |>
  tidyr::unnest_longer(col = coords) |>
  tidyr::separate_wider_delim(cols = coords, delim = ",", 
                              names = c("longitude", "latitude")) |>
  dplyr::mutate(longitude = as.numeric(longitude),
                latitude = as.numeric(latitude))

# Rename the place names to fit naming convention and remove images of 
# water slide park
osm_df <- osm_df |>
  dplyr::mutate(name = ifelse(is.na(name), "Unnamed_Rapids", name),
                name = stringr::str_replace_all(name, " ", "-"),
                name = stringr::str_replace_all(name, "\\(", ""),
                name = stringr::str_replace_all(name, "\\)", ""),
                name = stringr::str_replace_all(name, "\\+", ""),
                name = stringr::str_replace_all(name, "\\'", ""),
                name = stringr::str_replace_all(name, '\\"', "")) |>
  dplyr::mutate(name = dplyr::case_when(
    name == "29th-Street-Rapid" ~ "Twenty-Ninth-Street-Rapid",
    name == "45-Dam-Rapids" ~ "Fourty-Five-Dam-Rapids",
    stingr::str_detect(name, "^q") ~ stringr::str_replace(name, "^q", "Q"),
    .default = name
  )) |>
  dplyr::filter(!(name == "Rapids"))
