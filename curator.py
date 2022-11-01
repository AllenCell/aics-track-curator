import numpy as np
import pandas as pd
from pathlib import Path

class TrackNeighsCalculator():

    nneighs = 16

    def __init__(self, df):
        self.df = df
        self.calculate_n_nearest_tracks()

    def calculate_n_nearest_tracks(self):
        dist = self.calculate_tracks_distance_matrix()
        neigh_tracks = dist.argsort(axis=1)
        neigh_tracks = pd.DataFrame(neigh_tracks)
        neigh_tracks = neigh_tracks.set_index([0])
        neigh_tracks.index = neigh_tracks.index.rename("track")
        neigh_tracks.columns = [f"Neigh{i}" for i in neigh_tracks.columns]
        self.neigh_tracks = neigh_tracks

    def calculate_tracks_distance_matrix(self):
        dists = []
        df_ends = self.df.groupby("track_id", sort=True).agg(["first", "last"])
        for type1, type2 in zip(["first", "last", "first", "last"], ["first", "last", "last", "first"]):
            coords_type = []
            for ag in [type1, type2]:
                coords = df_ends[[("centroid_x", ag), ("centroid_y", ag), ("index_sequence", ag)]].values
                coords_type.append(coords)
            dist = np.linalg.norm(coords_type[0][:,None]-coords_type[1][None,:], axis=-1)
            dists.append(dist)
        dists = np.array(dists).min(axis=0)
        return dists

    def get_neighs_of_track(self, track):
        return self.neigh_tracks.loc[track].values[:self.nneighs].tolist()

class Lineage:

    def __init__(self):
        pass

    def set_data(self, df):
        self.df_full = df

    def check_df_consistency(self):
        for tid, df_track in self.df_full.groupby("track_id"):
            t0 = df_track.index_sequence.min()
            parent_id = df_track.parent_id.values[0]
            if parent_id > -1:
                t0_parent = self.df_full.loc[self.df_full.track_id==parent_id, "index_sequence"].min()
                if t0_parent > t0:
                    print(f"Inconsistency timing found between {tid} and {parent_id}. Dt = {t0-t0_parent}")
            noffs= self.df_full.loc[self.df_full.parent_id==tid, "track_id"].nunique()
            if noffs > 2:
                print(f"Track {tid} is assigned as parent of {noffs} tracks.")
        print("Dataframe consistency check complete." )

    def update(self, lineage_ids):
        self.nodes = pd.DataFrame({"track_id": lineage_ids})
        self.df = self.df_full.loc[self.df_full.track_id.isin(lineage_ids)]
        self.compute_generations()
        self.compute_edges()
        self.compute_node_position()

    def get_offspring_ids(self, track_id):
        return self.df.loc[self.df.parent_id==track_id].track_id.unique()

    def compute_generations(self):
        for nid, row in self.nodes.iterrows():
            self.nodes.at[nid, "term"] = self.df.loc[self.df.track_id==row.track_id, "termination"].max()
            self.nodes.at[nid, "t0"] = self.df.loc[self.df.track_id==row["track_id"], "index_sequence"].min()
            self.nodes.at[nid, "tf"] = self.df.loc[self.df.track_id==row["track_id"], "index_sequence"].max()
        self.nodes.term = self.nodes.term.astype(int)
        self.nodes["gen"] = 0
        self.nodes["up"] = 1
        self.nodes.at[self.nodes.t0.idxmin(), "gen"] = 1
        curr_gen = 1
        while self.nodes.gen.min() == 0:
            for nid, row in self.nodes.loc[self.nodes.gen==curr_gen].iterrows():
                offs = self.get_offspring_ids(row.track_id)
                self.nodes.loc[self.nodes.track_id.isin(offs), "gen"] = curr_gen+1
                for r, off in enumerate(offs):
                    self.nodes.loc[self.nodes.track_id==off, "up"] = int(2*(r-0.5))
            curr_gen += 1

    def t2idx(self, track_id):
        return self.nodes.loc[self.nodes.track_id==track_id].index[0]

    def idx2t(self, index):
        return self.nodes.at[index, "track_id"]

    def compute_node_position(self):
        self.nodes["y"] = np.nan
        index = self.nodes.t0.idxmin()
        self.nodes.at[index, "y"] = 0
        visited = []
        queue = [index]
        while queue:
            index = queue.pop(0)
            track_ids = self.get_offspring_ids(self.idx2t(index))
            for tid in track_ids:
                index_off = self.t2idx(tid)
                if index_off not in visited:
                    delta = 1.0 / np.power(2, self.nodes.at[index_off, "gen"]-1)
                    length = self.edges.query(f"source=={index}&target=={index_off}").length.values[0]
                    y = self.nodes.at[index, "y"] + delta*self.nodes.at[index_off, "up"]
                    self.nodes.at[index_off, "y"] = y
                    queue = [index_off]+queue
            visited.append(index)
        
    def compute_edges(self):
        edges = []
        for nid, row in self.nodes.iterrows():
            offs = self.get_offspring_ids(row.track_id)
            for track_id in offs:
                ti = self.nodes.at[nid, "t0"]
                tf = self.nodes.at[nid, "tf"]
                nidj = self.t2idx(track_id)
                edges.append({"source": nid, "target": nidj, "length": tf-ti})
        self.edges = pd.DataFrame(edges)
        self.edges.source = self.edges.source.astype(int)

    def add_lineage_to_plot(self, offset=(0, 0), axo=None):
        ax = axo
        if ax is None:
            fig, ax = plt.subplots()
        xo, yo = offset
        ax.scatter(xo+self.nodes.t0, yo+self.nodes.y, s=20, color="k")
        for index, row in self.nodes.iterrows():
            ax.annotate(int(row.track_id), (xo+row.t0, yo+row.y), xytext=(5, 5), textcoords="offset pixels")
            dx = len(self.df.loc[self.df.track_id==row.track_id])
            ax.plot([xo+row.t0, xo+row.tf], [yo+row.y, yo+row.y], "black")
            if row.term:
                marker = ["D", "P"][[1, 2].index(row.term)]
                color = ["magenta", "red"][[1, 2].index(row.term)]
                ax.scatter(xo+row.tf, yo+row.y, color=color, marker=marker, s=50, zorder=1000)
        for index, row in self.edges.iterrows():
            row_source = self.nodes.loc[row.source]
            row_target = self.nodes.loc[row.target]
            dx = len(self.df.loc[self.df.track_id==row_source.track_id])
            ax.plot([xo+row_source.tf, xo+row_source.tf], [yo+row_source.y, yo+row_target.y], "black")
        if axo is None:
            xmin, xmax = 0, self.df_full.index_sequence.max()
            fdx = 0.1*(xmax-xmin)
            ax.set_xlim(xmin-fdx, xmax+fdx)

    def get_lineage_dataset(self):
        data, used = {}, []
        for tid in self.df_full.track_id.unique():
            lineage = self.get_ids_in_same_lineage_as(self.df_full, tid)
            length = len(lineage)
            if length > 1:
                if lineage[0] not in used:
                    used += lineage
                    data[tid] = lineage
        return data

    def save_lineage_summary(self, path):
        w = self.df_full.index_sequence.max()
        data = self.get_lineage_dataset()
        n = round(np.sqrt(len(data)))
        fig, ax = plt.subplots(figsize=(4*n, 3*n))
        for tid, (track_id, lineage_ids) in enumerate(data.items()):
            x = tid%n
            y = tid//n
            self.update(lineage_ids)
            self.add_lineage_to_plot(offset=(x*(w+100), 2.5*y), axo=ax)
        plt.savefig(path, dpi=150)
        plt.close("all")

    def display_lineage_info(self, track_id):
        lineage_ids = self.get_ids_in_same_lineage_as(self.df_full, track_id)
        self.update(lineage_ids)
        idxmin = self.nodes.t0.idxmin()
        idxmax = self.nodes.tf.idxmax()
        print("Track", self.nodes.loc[idxmin, "track_id"], "starts at time", self.nodes.loc[idxmin, "t0"])
        for index, row in self.edges[::-1].iterrows():
            source = int(row.source)
            target = int(row.target)
            tidi = self.nodes.at[source, "track_id"]
            tidj = self.nodes.at[target, "track_id"]
            term = self.nodes.at[target, "term"]
            if not term:
                if len(self.edges.loc[self.edges.source==target]) == 2:
                    term = 3
                if self.nodes.at[target, "tf"] == 569:
                    term = 4
            term = [".", "E", "A", "D", "T"][term]
            print(tidi, "> \t",tidj, "\t", term)

    @staticmethod
    def get_parent(df, track_id):
        parent_id = df.loc[df.track_id==track_id].parent_id.unique()
        parent_id = [i for i in parent_id if i != -1]
        return parent_id

    @staticmethod
    def get_offsprings(df, track_id):
        offsprings = df.loc[df.parent_id==track_id].track_id.unique()
        return offsprings.tolist()

    @staticmethod
    def get_relatives(df, track_id):
        track_ids = Lineage.get_parent(df, track_id) + Lineage.get_offsprings(df, track_id)
        return track_ids

    @staticmethod
    def get_ids_in_same_lineage_as(df, track_id):
        lineage = [track_id]
        old, new = 0, 1
        while old != new:
            pool = []
            for track_id in lineage:
                pool += Lineage.get_relatives(df, track_id)
            lineage = np.unique(lineage+pool).tolist()
            old = new
            new = len(lineage)
        return lineage

class Curator:

    def __init__(self):
        self.view1 = GetActiveViewOrCreate("RenderView")
        self.reload()
        self.interpreter()

    def load_manifest(self):
        df = pd.read_csv(self.path/"manifest.csv", index_col=0)
        print(f"{df.track_id.nunique()} tracks loaded.")
        return df

    def reload(self):
        config = FindSource("config")
        if config is None:
            return
        path = str(sm.Fetch(config).GetValue(0, 1))
        self.path = Path(path.replace("\"",""))
        self.df = self.load_manifest()
        self.neighs_calculator = TrackNeighsCalculator(self.df)
        self.tracks = FindSource("tracks")
        self.movie = FindSource("movie")
        self.track = FindSource("ExtractSelection1")
        self.neighs = FindSource("ExtractSelection2")
        self.relatives = FindSource("ExtractSelection3")

    def initialize(self, text):
        path = Path(self.get_last_parameter(text))
        tracks = LegacyVTKReader(registrationName="tracks", FileNames=[str(path/"tracks.vtk")])
        tracks_display = Show(tracks, self.view1, "GeometryRepresentation")
        movie = TIFFSeriesReader(registrationName="movie", FileNames=[str(path/"movie.tif")])
        movie_display = Show(movie, self.view1, "UniformGridRepresentation")
        movie_display.SetRepresentationType('Slice')
        lut = GetColorTransferFunction('TiffScalars')
        lut.ApplyPreset('Grayscale', True)
        pd.DataFrame({"path": str(path)}, index=[0]).to_csv(str(path/"config.csv"))
        config = CSVReader(registrationName="config", FileName=[str(path/"config.csv")])
        self.view1.Update()

    def clear_workspace(self, text, delete_all=False):
        delete_all = False
        param = self.get_last_parameter(text)
        if param is not None:
            delete_all = True if "all" in param  else False
        selection = ["ExtractSelection1", "ExtractSelection2", "ExtractSelection3"]
        for key, value in GetSources().items():
            if delete_all | (key[0] in selection):
                Delete(value)
                del value

    def highlight_active_tracks(self, width, color):
        source = GetActiveSource()
        display = GetDisplayProperties(source, view=self.view1)
        display.DiffuseColor = color
        display.LineWidth = width

    def update_movie_view(self):
        bbox_track = np.array(self.track.GetDataInformation().GetBounds()).astype(int)
        GetDisplayProperties(self.movie, view=self.view1).Slice = bbox_track[-2]
        ResetCameraToDirection([1,1,1], [0,0,1], [0,-1,0])
        SetActiveSource(self.movie)
        Render()

    def load_track(self, text, update=True):
        self.clear_workspace("")
        ReloadFiles(self.tracks)

        track_id = int(self.get_last_parameter(text))
        parent_id = Lineage.get_parent(self.df, track_id)
        offspring_ids = Lineage.get_offsprings(self.df, track_id)
        relatives = parent_id + offspring_ids
        neighs = self.neighs_calculator.get_neighs_of_track(track_id)
        print(f"Track: {track_id}, parent: {parent_id}, offsprings: {offspring_ids}")

        for indexes, line_width, color in zip([[track_id], neighs, relatives], [10, 5, 8], [[1,0,0], [0,1,0], [1,1,0]]):
            if indexes:
                SetActiveSource(self.tracks)
                query = "|".join([f"(id=={t})" for t in indexes])
                QuerySelect(QueryString=query, FieldType="CELL")
                ExtractSelection()
                self.highlight_active_tracks(line_width, color)

        self.track = FindSource("ExtractSelection1")
        self.neighs = FindSource("ExtractSelection2")
        self.neighs = FindSource("ExtractSelection3")

        Hide(self.tracks, self.view1)
        Hide(self.neighs, self.view1)
        if update:
            self.update_movie_view()

    def get_current_track_id(self):
        track = sm.Fetch(self.track)
        track_id = int(vtknp.vtk_to_numpy(track.GetPointData().GetArray("track"))[0])
        return track_id

    def get_current_movie_tp(self):
        return GetDisplayProperties(self.movie, view=self.view1).Slice

    @staticmethod
    def get_polydata_from_df(df):

        cols = ["centroid_x", "centroid_y", "index_sequence"]
        coords = df[cols].values
        points = vtk.vtkPoints()
        points.SetData(vtknp.numpy_to_vtk(coords))

        scalar = vtk.vtkFloatArray()
        scalar.SetNumberOfComponents(1)
        scalar.SetName("track")
        scalar.SetNumberOfTuples(len(coords))
        scalar.FillComponent(0, 0)

        cells = vtk.vtkCellArray()
        for track_id, df_track in df.groupby("track_id", sort=True):
            polyLine = vtk.vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(len(df_track))
            for i, (index, row) in enumerate(df_track.iterrows()):
                polyLine.GetPointIds().SetId(i, index)
                scalar.SetTuple1(index, track_id)
            cells.InsertNextCell(polyLine)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(scalar)
        polydata.SetLines(cells)
        return polydata

    @staticmethod
    def save_polydata(polydata, path):
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(path)
        writer.Write()

    def convert_df_to_polydata_and_save(self):
        polydata = self.get_polydata_from_df(self.df)
        self.save_polydata(polydata, str(self.path/"tracks.vtk"))

    @staticmethod
    def sort_manifest_by_tracks_length(df):
        df_length = pd.DataFrame(df.groupby("track_id").size(), columns=["length"])
        df_length = df_length.sort_values(by="length", ascending=False)
        df_length = df_length.reset_index(drop=False)
        tracks_index_map = dict(zip(df_length.track_id, df_length.index))
        for col in ["track_id", "parent_id"]:
            df[col].replace(tracks_index_map, inplace=True)
        df = df.sort_values(by=["track_id", "index_sequence"])
        df = df.reset_index(drop=True)
        return df, tracks_index_map

    def save_manifest(self):
        self.df, tracks_index_map = self.sort_manifest_by_tracks_length(self.df)
        print(f"Saving {self.df.track_id.nunique()} tracks...")
        self.df.to_csv(self.path/"manifest.csv")
        self.convert_df_to_polydata_and_save()
        return tracks_index_map

    def split_track(self, text):
        tp = self.get_current_movie_tp()
        track_id = self.get_current_track_id()
        print(f"Spliting track {track_id} at timepoint {tp}...")

        track_id_new = self.df.track_id.max() + 1

        self.df.loc[(self.df.track_id==track_id)&(self.df.index_sequence>tp), "track_id"] = track_id_new
        self.df.loc[self.df.track_id==track_id_new, "parent_id"] = track_id if "unparent" not in text else -1
        # parent_id needs to be updated for any existing daughter track
        self.df.loc[self.df.parent_id==track_id, "parent_id"] = track_id_new
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id]}", update=False)

    def merge_track(self, text):
        tp = self.get_current_movie_tp()
        track_id = self.get_current_track_id()
        track_id_merge = int(self.get_last_parameter(text))

        t0 = self.df.loc[self.df.track_id==track_id_merge, "index_sequence"].min()
        tf = self.df.loc[self.df.track_id==track_id, "index_sequence"].max()
        if t0-tf < 0:
            print("Operation not permitted.")
            return

        print(f"Merging tracks {track_id} and {track_id_merge}...")

        parent_id = self.df.loc[(self.df.track_id==track_id)|(self.df.track_id==track_id_merge), "parent_id"].max()
        self.df.loc[self.df.track_id.isin([track_id, track_id_merge]), "parent_id"] = parent_id
        print(f"Parent id identified: {parent_id}")

        self.df.loc[self.df.track_id==track_id, "track_id"] = track_id_merge
        self.df.loc[self.df.parent_id==track_id, "parent_id"] = track_id_merge
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id_merge]}", update=False)

    def set_parent(self, text):
        track_id = self.get_current_track_id()
        track_id_parent = int(self.get_last_parameter(text))
        print(f"Setting track {track_id_parent} as parentof track {track_id}...")

        self.df.loc[self.df.track_id==track_id, "parent_id"] = track_id_parent
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id]}", update=False)

    def set_as_child_of(self, text):
        track_id = self.get_current_track_id()
        track_id_parent = int(self.get_last_parameter(text))
        print(f"Setting track {track_id_parent} as parent of track {track_id}...")
        self.df.loc[self.df.track_id==track_id, "parent_id"] = track_id_parent
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id]}", update=False)

    def set_as_parent_of(self, text):
        track_id = self.get_current_track_id()
        track_id_to_reload = track_id
        if ":" in text:
            track_id = int(text.split(" ")[1][:-1])
        track_id_childs = eval(self.get_last_parameter(text))
        print(f"Setting track {track_id} as parent of tracks {track_id_childs}...")

        for track_id_child in track_id_childs:
            self.df.loc[self.df.track_id==track_id_child, "parent_id"] = track_id
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id_to_reload]}", update=False)

    def get_track_info(self, text):
        track_id = int(self.get_last_parameter(text))
        df_track = self.df.loc[self.df.track_id==track_id]
        parent_id = Lineage.get_parent(self.df, track_id)
        offspring_ids = Lineage.get_offsprings(self.df, track_id)
        print(f"Track {track_id} has length: {len(df_track)}, parent {parent_id} and childs {offspring_ids}")
        lineage_ids = Lineage.get_ids_in_same_lineage_as(self.df, track_id)
        print(f"Full lineage: {lineage_ids}")

    @staticmethod
    def remove_short_tracks(df, min_length):
        df_length = df.groupby("track_id").size()
        df = df.loc[df.track_id.isin(df_length.loc[df_length>=min_length].index)]
        return df

    def import_dataset(self, text):
        self.clear_workspace("all")
        path = Path(self.get_last_parameter(text))

        df = pd.read_csv(path, index_col="CellId")
        df["centroid_x"] *= 0.4
        df["centroid_y"] *= 0.4
        df = self.remove_short_tracks(df, min_length=5)
        df["track_id_original"] = df["track_id"]
        df = df[["centroid_x", "centroid_y", "index_sequence", "track_id", "track_id_original"]]
        df["parent_id"] = -1
        df["termination"] = 0
        for f in df.columns:
            df[f] = df[f].astype(int)
        df, _ = self.sort_manifest_by_tracks_length(df)
        df.to_csv(path.parent/"manifest.csv")

        polydata = Curator.get_polydata_from_df(df)
        self.save_polydata(polydata, path.parent/"tracks.vtk")
        print(f"Imported dataframe with {df.track_id.nunique()} tracks.")

    def remesh(self, text):
        self.save_manifest()
        self.reload()
        ReloadFiles(self.tracks)

    def display_current_lineage_info(self, text):
        lineage = Lineage()
        lineage.set_data(self.df)
        track_id = self.get_current_track_id()
        lineage.display_lineage_info(track_id)

    def set_termination(self, text, mode):
        track_id = self.get_current_track_id()
        track_id_to_reload = track_id
        if ":" in text:
            track_id = int(text.split(":")[1])
        df_track = self.df.loc[self.df.track_id==track_id]
        term = 0
        if mode == "edge":
            term = 1
        if mode == "apoptosis":
            term = 2
        self.df.loc[self.df.track_id==track_id, "termination"] = term
        tracks_index_map = self.save_manifest()

        self.reload()
        self.load_track(f"load {tracks_index_map[track_id_to_reload]}", update=False)
        print(f"Setting termination {mode} for track {track_id}.")

    def progress(self, text):
        max_index = 569 # last timepoint of the movie, hard coded for goldilocks
        # calculate how many track_ids total for approach of interest
        # calculate how many track_ids have not been annotated 
        total_first_frame_list = []
        total_last_frame_list = []
        todo_first_frame_list = []
        todo_last_frame_list = []

        for track, df_track in self.df.groupby('track_id'):
            time = df_track['index_sequence'].to_numpy()

            if np.min(time) == 0:
                total_first_frame_list.append(track)
                x = track in self.df['parent_id'].unique()
                if x is False:
                    todo_first_frame_list.append(track)
            
            if np.max(time) == max_index:
                total_last_frame_list.append(track)
                parent = df_track['parent_id'].to_numpy()
                parent = parent[0]
                if parent == -1:
                    todo_last_frame_list.append(track)

        # calculate percent complete
        first = 100-((len(todo_first_frame_list))/(len(total_first_frame_list))*100)
        last = 100-((len(todo_last_frame_list))/(len(total_last_frame_list))*100)
        # display percent complete and next randomly chosen track_id from the to do list so that you can skip if necessary
        print('First Frame:', round(first),'% complete. \nNext track_id:', np.random.choice(todo_first_frame_list,1))
        print('Last Frame:', round(last),'% complete. \nNext track_id:', np.random.choice(todo_last_frame_list,1))

    def interpreter(self):
        text = input()
        if text.isnumeric():
            text = "load "+text
        if "cancel" in text:
            return
        if "help" in text:
            self.help()
        if "import" in text:
            self.import_dataset(text)
        if "init" in text:
            self.initialize(text)
        if "clear" in text:
            self.clear_workspace(text)
        if "load" in text:
            self.load_track(text)
        if "split" in text:
            self.split_track(text)
        if "merge" in text:
            self.merge_track(text)
        if "parentof" in text and "unparent" not in text:
            self.set_as_parent_of(text)
        if "childof" in text:
            self.set_as_child_of(text)
        if "info" in text:
            self.get_track_info(text)
        if "remesh" in text:
            self.remesh(text)
        if "l" == text:
            self.display_current_lineage_info(text)
        if "edge" in text:
           self.set_termination(text, "edge")
        if "apoptosis" in text:
            self.set_termination(text, "apoptosis")
        if "progress" in text:
            self.progress(text)
            

    def help(self):
        print("-----------------------------\n")
        print("cancel:\nAbort doing nothing.\n")
        print("import:\nStarts a new annotation project.\nUse: <import path/to/folder>.\n")
        print("init:\nInitialize a started project.\nUse: <init path/to/folder>.\n")
        print("load:\nLoad a new track for inspection.\nUse: <load track_id>.\n")
        print("clear:\nClear current view.\nUse: <clear>.\n")
        print("split:\nSplit a track in the current frame.\nUse: <split>.\n")
        print("merge:\nMerge current track with another track.\nUse: <merge track_id>.\n")
        print("parentof:\nSet the parent of a given track.\nUse either: <parentof [track_id1, track_id2, ...]> or <parentof track_id:[track_id1, track_id2, ...]>.\n")
        print("childof:\nSet the current track as child of a given track.\nUse: <childof track_id>.\n")
        print("info:\nDisplays information about a given track.\nUse: <info track_id>.\n")
        print("remesh:\nReloads the data.\nUse: <remesh>.\n")
        print("summary:\nGenerates PNG file with current annotated lineages.\nUse: <summary>.\n")
        print("edge:\n.Flags a given track as ending at the edge of the image.\nUse: <edge track_id>.\n")
        print("apoptosis:\nFlags a given track as ending in apoptotic event.\nUse: <apoptosis track_id>.\n")
        print("-----------------------------\n")

    @staticmethod
    def get_last_parameter(text):
        text = text.split(" ")
        if len(text) == 1:
            return None
        return text[-1]


if __name__ == "__vtkconsole__":

    import vtk
    from paraview.simple import *
    from paraview import servermanager as sm
    from vtk.util import numpy_support as vtknp

    curator = Curator()

if __name__ == "__main__":
    
    import sys
    import matplotlib.pyplot as plt
    print("Generating lineage.png...")

    path = sys.argv[1]
    df = pd.read_csv(f"{path}/manifest.csv", index_col=0)
    lineage = Lineage()
    lineage.set_data(df)
    lineage.check_df_consistency()
    lineage.save_lineage_summary(f"{path}/lineage.png")
    print("Lineage summary file saved.")
 
