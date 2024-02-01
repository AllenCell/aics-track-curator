# aics-track-curator
Python script for track curation in Paraview. Only available for OSX at the moment.

# One time actions:

1. Download paraview from https://www.paraview.org/download/.

2. Clone this repo:

`git clone https://github.com/aics-int/aics-track-curator.git`

3. Create an environment

```
conda create --name curator python=3.9
conda activate curator
bash install.sh
```

# Usage:

Open Paraview and you should see a button named `curator` in the menu. To check for the available actions, plese click `curator`, type `help` and hit OK.

To create a PNG image with the curated lineage information go to the terminal and type:

```
conda activate curator
python ~/.config/ParaView/Macros/curator.py path/to/dataset/directory
```

# How-to

## Method: 

Goal: Have a how-to document to provide detailed steps for generating lineage tracks

1) We want lineage information for all the cells in our movies. 
2) Why does a track end?
3) Validating track follows the cell of interest

### Start a new project:
1. See README.md on [Github Repo](https://github.com/aics-int/aics-track-curator) if this is your first project. 
2. You will need a local folder containing manifest.csv, curator.py, movie.tif
3. open ParaView
4. Curator: import path to manifest csv
5. Curator: init (path to folder)

### Continue a project:
1. open ParaView 
2. Curator: 
```
init /path/to/your/folder/
``` 
4. The [[Lineage Building Method (Curator)#Workflow|workflow]] saves the manifest as you go so you can start where you left off. Note: every update or reopening will reindex the track_ids from longest to shortest. [[Lineage Building Method (Curator)#Display Lineage|Visualize the lineage]] to continue a tree or start from the longest track (load 0))

### Set-up view windows:
1. add output messages to bottom pane, check the box "show full messages"
2. click +Z to center 
4. click 3D to toggle to 2D version
5. Pipeline browser
	- eyeball allows you to view tracks and movie in the viewer and spreadsheet
		- tracks represent centroids over time (only the tails are shown (aka prior in time frame not future))
	- Tracks = all tracks in movie (white)
	- ExtractSelection1 = track of interest you are labelling (red)
	- ExtractSelection2 = neighboring cells tracks (green)
	- ExtractSelection3 = lineage tracks (yellow)
8. click tracks, click split view, choose spreadsheet
	- now you should have the movie on the left, and spreadsheet on the right
	- keep the spreadsheet on the tracks dropdown
	- Tracks, ExtractSelection1, ExtractSelection2, and ExtractSelection3 can be chosen from the "showing" drop down
	- The green dashed square in the spreadsheet menu allows you to only view selected tracks 
9. Properties in left pane: Slicing -> Slice is how you move frames 
10. The green triangle tool is used to select track in the movie panel

![Paraview screenshot](Paraview.png)

### Curator Commands:
**init:** Initialize a started project.  Use: init path/to/folder.
**track_id**  Load a new track for inspection.  Use: track_id (i.e. 205)
**split:** Split a track in the current frame. If track continues to child Use: split. If track continues to incorrect cell Use: split unparent. If you are trying to merge with a track that starts at a frame too far before, load that track_id and use split unparent at the frame where the other track ends. 
**merge:** Merge current track with another track.  Use: merge track_id. Note: *tracks must me merged with tracks that start at the same frame as the last one that ends.* You will receive an error saying this operation cannot be performed if that is not the case. 
**parentof:** Set the parent of a given track.  Use either: parentof [track_id1, track_id2] or parentof track_id:[track_id1, track_id2]
**childof:** Set the current track as child of a given track.  Use: childof track_id
**edge:** Flags the loaded track as ending at the edge of the image.  Use: edge track_id
**apoptosis:** Flags the loaded track as ending in apoptotic event.  Use: apoptosis track_id
**remove:** Remove incorrect annotation of child track_id. Use: parentof -1: [track_id of wrong daughter] or move to the daughter track and Use: childof -1 
**l:** (lowercase L) Prints a text version of the lineage tree related to the track_id you are on. 
	Prints the start frame of the earliest track. if not 0, need to go back and continue the tree 
	"D" represents divides, and that both daughters have been annotated
	"T" terminates (by edge, or movie, or death), 
	"." designates the track still needs to be annotated.
**progress:** Prints % of cells complete in the first frame and last frame. Randomly selects a cell that is not annotated in first and last frame for you to choose to annotate. 


### Workflow:

1. Hotkey (shift + C) opens the curator plugin 
	*- only if you are clicked on the movie side, otherwise it will open a new render view* 
2. input: load track_id (ie. 0)
    - The program will automatically go to the slice the track starts in the movie
	- The manifest is organized from longest to shortest tracks starting with 0. 
	- Track IDs will update as you go so the same number will not always be the same track.
3. Go backwards in slice to find the parent cell and select tack using green triangle tool
	- click on or create a box around the track of interest
4. Above the spreadsheet the green square "show only selected elements", note the track_id
	- to select from the green tracks in the neighborhood choose ExtractSelection2 from the dropdown
	- to select from all the tracks available (if the cell you need is outside the neighborhood: choose tracks in the pipeline browser and from the spreadsheet dropdown.)
1. input: childof track_id
2. Click on the movie side, click on movie in pipeline browser, click on slice, pan through slice
3. split track if parent continues to child
	- split auto makes current track parentof new split track
	- if not use split unparent
4. get track_ids for children cells (check the output messages for most recent track_id as they can change with every command)
5. Load daughter cell of interest -> repeat the process
6. Record why a lineage ends: label as apoptosis or edge
7. When you get to the end of the movie, display the lineage (following the instructions below) to continue the tree in another location (i.e. visualize what track_id to go to). If the tree is complete load 0 in order to start another one. 

(d) - select points
this can be used to select more than one at a time! Nice!

(shift + A) - show all in pipeline browser

### Display Lineage:

To generate a figure with the lineage and current track_ids of the annotated cells at the time the figure is generated:

1. open terminal
2. conda activate curator
3. command below with path to folder
```
python ~/.config/ParaView/Macros/curator.py /path/to/your/folder/
``` 

i.e. 

![[lineage example.png|300]]

A) A complete lineage goes time from 0 to 568 (x-axis, units of time = frames)
B) Pink diamond (♦) means that the track goes to the edge of the FOV and terminates
C) Red plus sign (**+**) shows that the cell dies

---

### Additional Information

**Tips**

Tips to avoid ParaView application crashing
- When on the spreadsheet side, avoid selecting movie from the drop down menu
- This is because of the data types: tracks are polydata (points in a mesh) which are much smaller than movies which are a structured grid of data (points in 3D ie pixels) aka much larger and therefore hard for the application to load and it crashes. 
- Good news, if you do accidentally select movie while the green square is selected to only show selected elements, it likely wont crash!

Errors you might encounter:
- After init file, if you get a bunch of vtk errors, just close Paraview and start again. It should resolve. 
- After init file, you can ignore user warning: Could not import the lzma module. It doesn't seem to effect anything. 

Functions
- Edge and apoptosis annotations must be done on the currently loaded track. Future functionality could allow an unselected track to be annotated


**Q&A**

1.  Does the child ever not have a track? 
   Yes if its too far away. However merging to or selecting a closer track if available can potentially save the lineage. Also be mindful where you split the track to be close to the other daughter cell or both daughter cells so that they are within the neighborhood region. 

2. Is there an undo button? 
   No, however see the **remove** function under [[Lineage Building Method (Curator)#Curator Commands|curator commands]] for ways to correct wrongly assigned progeny. Edge and apoptosis annotations cannot currently be reversed. 

3. Is there a way to tell what tracks have already been annotated all the way through? 
   Currently, no. You can only visualize the current track, and in yellow its parent and child. You can visualize there is a list of all the track_ids using info. Visualizing the lineage tree and finding the one you are working on is the best way. If it goes from 0-568 frames you are done! 
   
4. Why did I get a track that looks like a scribble? 
   This happens when you merge with a track that that exists longer before the current frame used for merging. Basically, if you want to merge A with B, B has to start when A ends. If B starts many frames before, then you first have to go to B and split it in the frame where A ends. Additionally you cannot merge back in time! You must go to the prior track and then merge. Always go A to B *not* B to A.

** Paraview Troubleshooting**

How to make Paraview + Curator work on your Mac
- Lots of paraview versions crash upon opening on the M1 chip mac
- Download and open until you find a stable version
- Find latest version that loads fine on your computer. (for me that is version 5.11.0 )
- Scipy is a dependency of Curator. Paraview version 5.9.1 is the last version that had this built in. If that version does not load on your computer. 
- Right click on Paraview app and select 'show package contents' 
- Navigate to `/Applications/ParaView-5.11.0-RC1.app/Contents/Python`  
- Check if scipy is in the Python folder (maybe something has changed!)

How to get scipy on a new version of paraview:
-   NOTE copy pasting the scipy from version 5.9.1 doesnt work because it needs to be the matching python version
-   Create an environment with the matching python version to the Paraview you are using 
-   Go to environment we made with match python version and [create a path to the scipy](https://datacomy.com/python/anaconda/add_folder_to_path/)
-   `conda develop /PATH/TO/YOUR/FOLDER/WITH/MODULES`
-   copy the anaconda scipy folder (located here: `~/anaconda3/envs/YOUR_ENV/lib/pythonX.X/site-packages`)
-   Paste it into the paraview python folder!
