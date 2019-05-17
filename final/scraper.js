var scraper = require('google-play-scraper');
var d3 = require('d3-dsv');
var fs = require('fs');

var txt = fs.readFileSync('../apps.csv', 'utf8').toString();
// apps.csv from https://www.kaggle.com/orgesleka/android-apps#apps.csv

var data = d3.csvParse(txt);

// data.length == 403910

for (let i = 150500; i < 161000; ++i) {
    try {
        scraper.app({ appId: data[i].packageName }).then((app) => {
            fs.writeFile('app' + i + '.json', JSON.stringify(app), (err) => {
                if (err) throw err
            });
        }).catch((e) => { });
    } catch (e) { }
}