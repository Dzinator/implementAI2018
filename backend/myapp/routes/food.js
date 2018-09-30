var express = require('express');
var router = express.Router();
const { exec } = require('child_process');
var multer = require('multer');
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, '/home/amir/github/implementAI2018/backend/myapp/public/images/ocv/')
    },
    filename: function ( req, file, cb ) {
        cb( null, "ocv.jpg");
    }
});
const fs = require('fs');
const path = require('path');
var upload = multer( { storage: storage } );

/* GET users listing. */
router.get('/', function(req, res, next) {
    res.render('food', { title: 'Express' });
});

router.get('/clear', function(req, res, next) {
    var directory = "/home/amir/github/implementAI2018/backend/myapp/public/images/ocv/";
    fs.readdir(directory, (err, files) => {
        if (err) throw err;
        for (const file of files) {
            fs.unlinkSync(path.join(directory, file), err => {
                if (err) throw err;
            });
        }
    });
    res.redirect('/food');
});

router.post('/calories', upload.single('photo'), function(req, res, next) {
    if (!req.file) {
        return res.status(400).send('No files were uploaded.');
    }

    var directory = "/home/amir/github/implementAI2018/backend/myapp/public/images/ocv/";
    fs.readdir(directory, (err, files) => {
        if (err) throw err;

        for (const file of files) {
            if(file !== "ocv.jpg") {
                fs.unlinkSync(path.join(directory, file), err => {
                    if (err) throw err;
                });
            }
        }
    });

    exec('/home/amir/github/implementAI2018/opencv/cmake-build-debug/food-detection '
        + '"/home/amir/github/implementAI2018/backend/myapp/public/images/ocv/ocv.jpg" ' + req.body.kmeans + ' ' + req.file.originalname, (err, stdout, stderr) => {
        if (err) {
            return;
        }

        // the *entire* stdout and stderr (buffered)
        console.log(`stdout: ${stdout}`);
        console.log(`stderr: ${stderr}`);
        res.redirect('/food');
    });
});

module.exports = router;
