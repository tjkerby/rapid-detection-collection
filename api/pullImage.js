const axios = require('axios');
const fs = require('fs');

require('dotenv').config();

async function getSatelliteImage(latitude, longitude, zoom = 18) {
    const apiKey = process.env.GOOGLE_MAPS_API_KEY;
    
    if (!apiKey) {
        throw new Error('Google Maps API key not found in environment variables');
    }

    const baseUrl = 'https://maps.googleapis.com/maps/api/staticmap';
    const size = '640x640'; // Max size for free tier
    const mapType = 'satellite';

    const url = `${baseUrl}?center=${latitude},${longitude}&zoom=${zoom}&size=${size}&maptype=${mapType}&key=${apiKey}`;

    try {
        const response = await axios({
            method: 'get',
            url: url,
            responseType: 'arraybuffer'
        });

        return response.data;
    } catch (error) {
        console.error('Error fetching satellite image:', error.message);
        throw error;
    }
}

// Example usage
async function saveSatelliteImage(latitude, longitude, filename) {
    try {
        const imageData = await getSatelliteImage(latitude, longitude);
        fs.writeFileSync(filename, imageData);
        console.log(`Satellite image saved as ${filename}`);
    } catch (error) {
        console.error('Failed to save satellite image:', error);
    }
}

module.exports = {
    getSatelliteImage,
    saveSatelliteImage
};

// Example:
// saveSatelliteImage(40.7128, -74.0060, 'satellite-image.png');