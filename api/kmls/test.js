const tj = require("@tmcw/togeojson");
const fs = require("fs");
// node doesn't have xml parsing or a dom. use xmldom
const DOMParser = require("xmldom").DOMParser;

const kml = new DOMParser().parseFromString(fs.readFileSync("foo.kml", "utf8"));

const converted = tj.kml(kml);