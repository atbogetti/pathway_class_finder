# extract_success_ray.py
#
# Code that traces through an assign.h5 to generate a list of lists containing
#     successful source -> target transitions. 
#
# This version has ray parallelization.
#
# Main function includes:
#   * Output a list containing traces of source -> sink iter/seg numbers
#   * Output the trajectories of the traces
#   * Rewrite an h5 file so any non-successful trajectories have 0 weight.
#
# Defaults to the first passage (from basis state to first touch target)
#     for all of the above.
# Change `trace_basis` to False to trace the transition time.
#
# Written by Jeremy Leung
# Last modified: Nov 18th, 2022


import westpa.analysis as wa
import h5py
import pickle
import numpy
import ray
from tqdm.auto import trange
from shutil import copyfile
from os import mkdir
from os.path import isdir, exists


@ray.remote
class TraceActor:
    def __init__(self, assign_name, run_name):
        self.assign_file = h5py.File(assign_name, 'r')
        self.h5_run = wa.Run(run_name)

    def trace_seg_to_last_state(
        self,
        source_state_num,
        target_state_num,
        iteration_num,
        segment_num,
        first_iter,
        trace_basis,
        out_traj,
        out_traj_ext,
        out_state_ext,
        out_top,
        hdf5,
        #tqdm_bar,
    ):
        """
        Code that traces a seg to frame it leaves source state. Can run to export trajectories too!
    
        Returns
        =======
    
        indv_trace : lst of lst
            A list of list containing the iteration and segment numbers of the trace. None if the trajectory does not end in the target state.
    
        indv_traj : obj
            A BasicMDTrajectory() or HDF5MDTrajectory() object or None. Basically a subclass of the MDTraj Trajectory object.
    
        frame_info : lst of lst
            A list of list containing the frame number of each iteration. Goes backwards from the last frame. Returns None if does not end in the target state.
    
        """
        run = self.h5_run
        assign_file = self.assign_file
        indv_trace = []
        trace = run.iteration(iteration_num).walker(segment_num).trace()
        traj_len = (
            len(run.iteration(iteration_num).walker(segment_num).pcoords) - 1
        )  # Length is number of frames in traj + 1 (parent); only caring about the number of frames
    
        #tqdm_bar.set_description(f"tracing {iteration_num}.{segment_num}")
        # Going through segs in reverse order
        for iwalker in reversed(trace):
            pc_arr = iwalker.pcoords[:,0][-1]
            pc_arr2 = iwalker.pcoords[:,1][-1]
            weight = iwalker.weight
            corr_assign = assign_file["statelabels"][
                iwalker.iteration.summary.name - 1, iwalker.segment_summary.name
            ]
            if iwalker.iteration.summary.name == iteration_num:
                # Dealing with cases where we're in the first iteration looked at
                term_frame_num = numpy.where(corr_assign == target_state_num)[0][0]  # Taking only the first instance in tstate
                if trace_basis is False:
                    if source_state_num in corr_assign[: term_frame_num + 1]:
                        # Went from source to target in one iteration. neat.
                        source_frame_num = numpy.where(corr_assign == source_state_num)[0][-1]
                        indv_trace.append([iteration_num, segment_num, pc_arr, pc_arr2, weight])
                        break
                # Just a normal iteration where we reached target state and that's it.
                indv_trace.append([iteration_num, segment_num,  pc_arr, pc_arr2, weight])
    
            elif iwalker.iteration.summary.name != iteration_num:
                # Dealing with cases where we're in other iterations
                if source_state_num not in corr_assign:
                    # The segment did not visit the source this iteration.
                    if iwalker.iteration.summary.name != iteration_num:
                        if target_state_num in corr_assign:
                            # Hey, there is a target > target transition.
                            # Breaking out...
                            #run.close()
                            return None, None, None
                        else:
                            # If haven't been in state A, and not in target iteration... add the whole iteration into list
                            # Also making sure it's not a target -> target transition
                            indv_trace.append(
                                [iwalker.iteration.summary.name, iwalker.segment_summary.name, pc_arr, pc_arr2, weight]
                            )
                else:
                    # This else captures cases where it was in the source in this iteration
                    # Looking for the last frame it was in source state
                    source_frame_num = numpy.where(corr_assign == source_state_num)[0][-1]
                    if target_state_num in corr_assign[source_frame_num:]:
                        # Catching the ouchie case where it went from source -> target -> target. Woopsies!
                        #run.close()
                        return None, None, None
                    else:
                        # Final case where it's definitely source -> target
                        indv_trace.append(
                            [iwalker.iteration.summary.name, iwalker.segment_summary.name, pc_arr, pc_arr2, weight]
                        )
                        if trace_basis is False:
                            break
    
        frame_info = []
        try:
            source_frame_num
        except NameError:
            source_frame_num = 0
        start_trace = (traj_len - source_frame_num) + ((len(indv_trace) - 1) * traj_len)  # Total number of frames necessary
        end_trace = traj_len - term_frame_num
        frame_info.append(term_frame_num)
        for i in range(len(indv_trace) - 1, 1, -1):
            frame_info.append(traj_len)
        frame_info.append(source_frame_num)
    
        # Block for outputting the traj
        if out_traj:
            #tqdm_bar.set_description(f"outputting traj for {iteration_num}.{segment_num}")
            if hdf5 is False:
                trajectory = wa.BasicMDTrajectory(
                    traj_ext=out_traj_ext, state_ext=out_state_ext, top=out_top
                )
            elif hdf5 is True:
                trajectory = wa.HDF5MDTrajectory()
            else:
                print("unable to output trajectory")
                return indv_trace, None, frame_info
    
            # Extra check so it won't output past first_iter. 
            #if first_iter != 1:
            #    max_length = iter_num - first_iter + 1
            #    trace = run.iteration(iteration_num).walker(segment_num).trace(max_length=max_length)

            indv_traj = trajectory(trace)
            if trace_basis is False:
                indv_traj = indv_traj[-start_trace:-end_trace]
        else:
            indv_traj = None
    
        #print(indv_trace)
    
        #run.close()
        return indv_trace, indv_traj, frame_info
   

def retain_succ(
    west_name="west.h5",
    assign_name="assign.h5",
    source_state_num=0,
    target_state_num=1,
    first_iter=1,
    last_iter=None,
    trace_basis=True,
    out_traj=False,
    out_traj_ext=".nc",
    out_state_ext="_nowat.ncrst",
    out_top="bdpa_wsh2045_nowat.prmtop",
    out_dir="succ_traj",
    hdf5=False,
    rewrite_weights=False,
    threads=None,
):
    """
    Code that goes through an assign file (assign_name) and extracts iteration
    and segment number + its trace into a pickle object. Defaults to just
    the whole passage but can be override by trace_basis=False
    to just the transition time (between it last exits source to when
    it first touches target). Can also extract the trajectories along
    the way with out_traj=True.

    Arguments
    =========
    west_name : str, default: "west.h5"
        Name of the HDF5 File.

    assign_name : str, default: "assign.h5"
        Name of the `w_assign` output file.

    source_state_num : int, default: 0
        Index of the source state. Should be the first state defined in
        west.cfg before running `w_assign`.

    sink_state_num : int, default: 1
        Index of the sink state. Should be the second state defined in
        west.cfg before running `w_assign`.

    first_iter : int, default: 1
        First iteration to analyze. Inclusive.

    last_iter : int, default: None
        Last iteration to analyze. Inclusive. None implies it will
        analyze all labeled iterations.

    trace_basis : bool, default: True
        Option to analyze each successful trajectory up till its
        basis state. False will only analyze the transition time
        (i.e. last it exited source until it first touches the target state).

    out_traj : bool, default: False
        Option to output trajectory files into `out_dir`.

    out_traj_ext : str, default: ".nc"
        Extension of the segment files. The name of the file is assumed to be
        `seg`, meaning the default name of the file is `seg.nc`.

    out_state_ext : str, default: "_nowat.ncrst"
        Extension of the restart files. The name of the file is assumed to be
        `seg`, meaning the default name the file is `seg_nowat.ncrst`.

    out_top : str, default: "bdpa_wsh2045_nowat.prmtop"
        Name of the topology file.
        Name is relative to `$WEST_SIM_ROOT/common_files/`.

    out_dir : str, default: "succ_traj"
        Name of directory to output the trajectories.
        Name is relative to `$WEST_SIM_ROOT`.

    hdf5 : bool, default: False
        Option to use the `HDF5MDTrajectory()` object in `westpa.analysis`
        instead. To be used with trajectories saved with the HDF5 Framework.

    rewrite_weights : bool, default: False
        Option to zero out the weights of all segments that are not part of
        the successful trajectories ensemble. Note this generates a new h5
        file with the _succ suffix added. Default name is thus `west_succ.h5`.

    threads : int, default: None
        Number of actors/workers for ray to initialize. It will take over all
        CPUs if None.

    Outputs
    =======
    'output.pickle': pickle obj
        A list of the form [n_traj, n_frame, [n_iter, n_seg]]. Note that each
        list runs backwards from the target iteration.

    'frame_info.pickle': pickle obj
        A list of the form [n_traj, [index number of each segment]]. Note that
        each list runs backwards from the target iteration.

    """
    # Variables validation
    if last_iter is None:
        with h5py.File(assign_name, "r") as assign_file:
            last_iter = len(assign_file["nsegs"])
    else:
        assert type(last_iter) is int, "last_iter is not legal and must be int."

    # Create Output file
    try:    
        mkdir(out_dir)
    except FileExistsError:
        print(f"Folder {out_dir} already exists. Files within might be overwritten.")

    # Copying the file
    name_root = west_name.rsplit(".h5", maxsplit=1)[0]
    new_file = f"{out_dir}/{name_root}_succ.h5"
    if not exists(new_file):
        copyfile(west_name, new_file)

    # Prepping final list to be outputted
    trace_out_list = []
    frame_info_list = []

    # Ray stuff...
    ray.init()
    if threads is None:
        n_actors = int(ray.available_resources().get("CPU",1))
    else:
        n_actors = threads
    all_ray_actors = []
    for i in range(n_actors):
        all_ray_actors.append(TraceActor.remote(assign_name, new_file))
    
    # Yes, tracing backwards from the last iteration. This will (theoretically) allow us to catch duplicates more efficiently.
    with h5py.File(assign_name, "r") as assign_file, wa.Run(new_file) as h5_run:
        tqdm_iter = trange(last_iter, first_iter - 1, -1, desc="iter")
        for n_iter in tqdm_iter:
            all_ray_tasks = []
            for n_seg in range(assign_file["nsegs"][n_iter - 1]):
                flag = False
                if target_state_num in assign_file["statelabels"][n_iter - 1, n_seg]:
                    # For some reason not really working...
                    for array in trace_out_list:
                        # print([n_iter,n_seg])
                        if [n_iter, n_seg] in array:
                            # Skip to next segment... It's already been captured
                            flag = True
                            print("overlap")
                            break
                    if flag is True:
                        continue
                    else:
                        all_ray_tasks.append(all_ray_actors[n_seg%n_actors].trace_seg_to_last_state.remote(
                            source_state_num,
                            target_state_num,
                            n_iter,
                            n_seg,
                            first_iter,
                            trace_basis,
                            out_traj,
                            out_traj_ext,
                            out_state_ext,
                            out_top,
                            hdf5,
                            ))

            while all_ray_tasks:
                finished, all_ray_tasks = ray.wait(all_ray_tasks, num_returns=min(n_actors*5, len(all_ray_tasks)))
                results = ray.get(finished)
                for each_return in results:
                    (trace_output, traj_output, frame_info) = each_return
                    if trace_output is not None:
                        trace_out_list.append(trace_output)
                    if traj_output is not None:
                        traj_output.save(f"{out_dir}/{n_iter}_{n_seg}{out_traj_ext}")
                    if frame_info is not None:
                        frame_info_list.append(frame_info)

    # Output list
    trace_out_list = sorted(trace_out_list, key=lambda x: (-x[0][0], x[0][1]))
    with open(f"{out_dir}/output.pickle", "wb") as fo:
        pickle.dump(trace_out_list, fo)

    with open(f"{out_dir}/frame_info.pickle", "wb") as fo:
        pickle.dump(frame_info_list, fo)

    # Finally, zero out (iter,seg) that do not fall in this "successful" list.
    if rewrite_weights:
        exclusive_set = {tuple(pair) for list in trace_out_list for pair in list}
        with h5py.File(new_file, "r+") as h5file:
            for n_iter in tqdm_iter:
                for n_seg in range(assign_file["nsegs"][n_iter - 1]):
                    if (n_iter, n_seg) not in exclusive_set:
                        h5file[f"iterations/iter_{n_iter:>08}/seg_index"]["weight", n_seg] = 0


if __name__ == "__main__":
    retain_succ(
        west_name="../west.h5",
        assign_name="assign.h5",
        source_state_num=0,
        target_state_num=1,
        first_iter=1,
        last_iter=None,
        trace_basis=False,
        out_traj=False,
        out_traj_ext=".nc",
        out_state_ext="_img.ncrst",
        out_top="cat10.prmtop",
        out_dir="succ_traj",
        hdf5=False,
        rewrite_weights=False,
   )
