// using togeojson in nodejs

const tj = require("@tmcw/togeojson");
const fs = require("fs");
const path = require("path");
const DOMParser = require("xmldom").DOMParser;

// Get input path from command line arguments
const inputPath = process.argv[2];
const outputFile = process.argv[3] || 'out/output.json';

if (!inputPath) {
  console.error('Please provide an input KML file or directory path');
  process.exit(1);
}

try {
  let allFeatures = [];

  if (fs.lstatSync(inputPath).isDirectory()) {
    // Process all KML files in directory
    const files = fs.readdirSync(inputPath)
      .filter(file => file.toLowerCase().endsWith('.kml'))
      .map(file => path.join(inputPath, file));
    
    files.forEach(file => {
      const rawKml = fs.readFileSync(file, "utf8").replace(/ns0:/g, '');
      const kml = new DOMParser().parseFromString(rawKml, "text/xml");
      const converted = tj.kml(kml);
      allFeatures = allFeatures.concat(converted.features || []);
    });
  } else {
    // Process single file
    const rawKml = fs.readFileSync(inputPath, "utf8").replace(/ns0:/g, '');
    const kml = new DOMParser().parseFromString(rawKml, "text/xml");
    const converted = tj.kml(kml);
    allFeatures = allFeatures.concat(converted.features || []);
  }
  
  console.log(`Total features extracted: ${allFeatures.length}`);
  
  // Write output
  const writable = fs.createWriteStream(outputFile);
  writable.write('{\n  "type": "FeatureCollection",\n  "features": [\n');
  allFeatures.forEach((feature, i) => {
    if (feature.properties && feature.properties.description && feature.properties.description.value) {
      feature.properties.description.value = '';
    }
    writable.write(JSON.stringify(feature, null, 2));
    if (i < allFeatures.length - 1) {
      writable.write(',\n');
    }
  });
  writable.write('\n]\n}');
  writable.end();
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}