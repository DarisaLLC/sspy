# sspy
	
Implementation of "Method and apparatus for acquisition, compression, and characterization of spatiotemporal signals"

Covered by U.S. Patents 9,001,884 and 7,672,369 and Patents in EU and Japan. 

usage: main.py [-h] --content CONTENT [--match MATCH] [--outpath OUTPATH]
               --duration DURATION [--show SHOW] [--write WRITE]
               [--prefix PREFIX] --type TYPE [--levelset LEVELSET]
               [--voxels VOXELS] [--channel CHANNEL]

SelfSimilarator

optional arguments:
  -h, --help            show this help message and exit
  
  --content CONTENT, -i CONTENT
                        Directory of sequentially numbered image files or TIF
                        multipage file
                        
  --match MATCH, -m MATCH
                        0 squared_ncv, 1 variation_of_information
                        
  --outpath OUTPATH, -o OUTPATH
                        Path of output dir
                        
  --duration DURATION, -d DURATION
                        Moving Temporal Window Size or -1 For All
                        
  --show SHOW, -s SHOW
  
  --write WRITE, -w WRITE
  
  --prefix PREFIX, -p PREFIX
                        Image File Prefix, i.e. prefix0001.png
                        
  --type TYPE, -t TYPE  Image File Extension
  
  --levelset LEVELSET, -l LEVELSET **EXPERIMENTAL **
                        Perform level setting with 1 / l fractions
                        
  --voxels VOXELS, -v VOXELS **EXPERIMENTAL **
                        Use voxels as sequence data
                        
  --channel CHANNEL, -c CHANNEL
                        Channel to use if multichannel default converts to
                        gray
