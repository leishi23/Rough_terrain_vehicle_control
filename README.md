

<h1 align="center">
  <br>
  <a href="https://carla.readthedocs.io/en/latest/"><img src="https://i.ytimg.com/vi/AZhzZy57XeU/maxresdefault.jpg" alt="Dexterous Manipulation Online Planning Repo" width="400", height="200"></a>
  <br>
  CARLA Simulation based on RoadRunner
  <br>
</h1>

<!-- <h4 align="center">A minimal Markdown Editor desktop app built on top of <a href="http://electron.atom.io" target="_blank">Electron</a>.</h4> -->

<p align="center">
  <a href="https://github.com/leishi23"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#Contents">Contents</a> 
</p>

![screenshot](https://auto.aicurious.io/autonomous.gif)

## Key Features

* Customized map on RoadRunner with high freedom
* Self built synchronous dataset from simulation
* Rewrite [Tensorflow model](https://github.com/gkahn13/badgr) with PyTorch for better computation velocity

## Contents

- **data_collection.py** is to collect raw data from CARLA
  - Customized a map on [RoadRunner](https://www.mathworks.com/products/roadrunner.html) and export it into _Linux built CARLA_.
  - Run `make import` to import map. [(tutorial)](https://carla.readthedocs.io/en/latest/tuto_M_add_map_source/)
  - Set parameters like map/vehicle selection, vehicle motion params(velocity, etc), sensor params and number of datapoints to collect. 
  - Run the `data_collection.py` to collect and store raw data into `json/csv` files.

- **random_dataset.py** is to build randomized synchronomous training dataset and test dataset from raw data
  - Function: Delete remains datapoints and some images to prepare for the dataset procession. Merge 9 time steps(raw data) into 1 data point.
  - Time steps every datapoint: 9
  - Data structure: ground truth/ action input. 

- **dataset_check.py** is to check the collected raw data like velocity and generate corresponding image

- **eval.py** is the MPPI Control evaluation 
  - Run CARLA to get sensor data 
  - Then export these data into model to generate motion prediction. 
  - With motion prediction and reward function, get an optimal path (angular velocity and linear velocity list)
  - Move vehicle in CARLA based on optimal path.
  - Get new sensor data, repeat steps above.

- **model.py** is for the CNN-LSTM model
![image](https://lh6.googleusercontent.com/uVKMBsiDiva5-peH38zoXfd89Ss-fYqzyttcFD0s1R04egYsM6VLQYCzQ9YsVs4VkiS_CAXHPIgPVJl3B2Lzko0ZOy_tBi8Xw6G8TBHvYMIZlGcA=w1280)
(from baseline project [BADGR](https://sites.google.com/view/badgr))

- **trainer.py** is for training

- **real_time_plot.py** is for real time data plot, like RGB image and lidar info.
  - Launch the CARLA simulator. 
  - Adjust ego-vehicle into auto-driving mode.
  - Run `real_time_plot.py` to generate real time plot of multiple sensors.

> **_Note_**: This project is actually unfinished. The evaluation part still has something wrong. But limited by my PC hardware and my PhD program isn't about mobile robot, this project stops here.


<!-- ## Download

You can [download](https://github.com/amitmerchant1990/electron-markdownify/releases/tag/v1.2.0) the latest installable version of Markdownify for Windows, macOS and Linux.

## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify

## Support

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a> -->

<!-- ## You may also like...

- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp;
> GitHub [@amitmerchant1990](https://github.com/amitmerchant1990) &nbsp;&middot;&nbsp;
> Twitter [@amit_merchant](https://twitter.com/amit_merchant) -->

