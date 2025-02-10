// using togeojson in nodejs

const tj = require("@tmcw/togeojson");
const fs = require("fs");
// node doesn't have xml parsing or a dom. use xmldom
const DOMParser = require("xmldom").DOMParser;

const kml = new DOMParser().parseFromString(fs.readFileSync("in/kml_test.kml", "utf8"));

const converted = tj.kml(kml);
const features = converted.features || [];
const writable = fs.createWriteStream('out/output.json');

writable.write('{\n  "type": "FeatureCollection",\n  "features": [\n');
features.forEach((feature, i) => {
  writable.write(JSON.stringify(feature, null, 2));
  if (i < features.length - 1) {
    writable.write(',\n');
  }
});
writable.write('\n]\n}');
writable.end();