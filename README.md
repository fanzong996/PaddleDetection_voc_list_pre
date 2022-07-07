# PaddleDetection_voc_list_pre
## You Should Know:
This tool is used for paddledetection
When training the voc dataset, the yml file(configs) lacks necessary trainval.txt and test.txt
The tool is built for it!
## one more thing
when using AI Studio to run the module, the jpg or xml file dir in voc list should be divided by ' / ',
however using lacal terminal, it divided by ' \ '
## setp1: run create_txt.py
## setp2: run txt_write.py
## run preview:
000019
000061
000090
000112
000144
000196
000240
000250
000263
000272
000276
000298
000353
000356
000365
--step1
VOCdevkit\VOC2012\JPEGImages\000019.jpg VOCdevkit\VOC2012\Annotations\000019.xml
VOCdevkit\VOC2012\JPEGImages\000061.jpg VOCdevkit\VOC2012\Annotations\000061.xml
VOCdevkit\VOC2012\JPEGImages\000090.jpg VOCdevkit\VOC2012\Annotations\000090.xml
VOCdevkit\VOC2012\JPEGImages\000112.jpg VOCdevkit\VOC2012\Annotations\000112.xml
VOCdevkit\VOC2012\JPEGImages\000144.jpg VOCdevkit\VOC2012\Annotations\000144.xml
VOCdevkit\VOC2012\JPEGImages\000196.jpg VOCdevkit\VOC2012\Annotations\000196.xml
VOCdevkit\VOC2012\JPEGImages\000240.jpg VOCdevkit\VOC2012\Annotations\000240.xml
VOCdevkit\VOC2012\JPEGImages\000250.jpg VOCdevkit\VOC2012\Annotations\000250.xml
--step2
