const fs = require("fs");

exports.readFile = (fileName) => JSON.parse(fs.readFileSync(fileName, "utf8"));
