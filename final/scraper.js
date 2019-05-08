var scraper = require('google-play-scraper');
var d3 = require('d3-dsv');
var fs = require('fs');

var txt = fs.readFileSync('apps.csv', 'utf8').toString();

var data = d3.csvParse(txt);

ids = [];

for (let i = 0; i < data.length; ++i) {
    ids.push(data[i].packageName);
}

console.log(ids);