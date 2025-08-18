# This function implements key signatures required for the Google Maps APIs. 
# input_url: A character vector of length one containing a URL built by the 
# code in par_script to download a static image
# secret: A character vector of length one containing a Maps API secret 
# obtained from the Google Cloud dashboard
# Returns: The input URL with the signature appended
sign_url <- function(input_url, secret) {

  # Decode the secret key from Base64
  decoded_key <- base64enc::base64decode(secret)

  # Create HMAC SHA-1 signature
  signature <- digest::hmac(decoded_key, input_url, algo = "sha1", raw = TRUE)

  # Encode the binary signature in URL-safe Base64
  encoded_signature <- base64enc::base64encode(signature)

  encoded_signature <- gsub("\\+", "-", gsub("/", "_", encoded_signature))
  
  # Reconstruct the original URL and append the signature
  signed_url <- paste0(input_url, "&signature=", encoded_signature)
  
  return(signed_url)
}
