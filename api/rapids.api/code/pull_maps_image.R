# Function to pull an image from the Google Maps Static API
# given the latitude and longitude and valid Maps API credentials
# title: The name of the river or watershed of the image location
# latitude: The latitude coordinate of the image
# longitude: The longitude coordinate of the image
# zoom: The zoom level of the image
# api_key: A Google Maps API key with permissions to access the Maps Static API
# secret: A Maps API secret obtained from the Google Cloud dashboard
# Returns: A data.frame with one row with the following fields: 
#   image: A primary key identifying each image constructed from the title, 
#   latitude, longitude, and zoom fields
#   latitude: The latitude coordinate of the image
#   longitude: The longitude coordinate of the image
#   zoom: The zoom level of the image
#   timestamp: A Unix timestamp indicating when the image was downloaded.
pull_maps_image <- function(title, latitude, longitude, zoom, api_key, secret, 
                            image_dir) {
  # Construct the URL for Google Maps Static API
  base_url <- "https://maps.googleapis.com/maps/api/staticmap"
  params <- list(
    center =  paste0(latitude, ",", longitude),
    zoom = zoom,
    size = "640x640",
    maptype = "satellite",
    key = api_key,
    scale = 2
  )
  
  url_path <- httr::modify_url("/maps/api/staticmap", query = params) 
  url_path <- stringr::str_remove(url_path, "^://")
  
  base_url_signed <- sign_url(url_path, secret = secret)
  
  # Make the request
  response <- httr::GET(url = paste0("https://maps.googleapis.com",
                                     base_url_signed))
  time_stamp <- as.numeric(lubridate::now())
  
  if (as.numeric(response$status_code) == 200) {
    longitude_char <- stringr::str_replace(longitude, "\\.", "~")
    latitude_char <- stringr::str_replace(latitude, "\\.", "~")
    filename <- paste0(title, "_", longitude_char, "_", latitude_char, 
                       "_z", zoom)
    
    raw_content <- httr::content(response, "raw")
    
    writeBin(raw_content, paste0(image_dir, filename, ".jpg"))
    
    # Create a list with data
    image_data <- data.frame(
      image = filename,
      name = title,
      longitude = longitude,
      latitude = latitude,
      zoom = zoom,
      timestamp = time_stamp
    )
    return(image_data)
  } else {
    print(sprintf("Request failed with status: %d", 
                  httr::http_status(response)$message))
    return(NULL)
  }
}
