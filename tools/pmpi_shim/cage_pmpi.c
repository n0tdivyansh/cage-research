/*
 * CAGE PMPI shim — records MPI communication for CAGE (cage.extract.pmpi).
 *
 * STATUS: UNTESTED. Written against the MPI-3 C bindings; compile and run the self-test
 * (tools/pmpi_shim/selftest.c, `make test`) on a machine with MPI before trusting it.
 *
 * Build:   make            -> libcagepmpi.so
 * Use:     CAGE_OUT=traces/run1 LD_PRELOAD=$PWD/libcagepmpi.so mpirun -np 16 ./app
 *          (or link with -L. -lcagepmpi before -lmpi)
 * Steps:   call MPI_Pcontrol(1) at each iteration boundary in the app; the shim increments
 *          its step counter. Apps that never call it produce step 0 for everything.
 *
 * Output:  $CAGE_OUT/cage_trace.<world_rank>.csv
 *          kind,op,src,dst,bytes,step,t0,t1,root,members
 *   p2p  recorded by the sender only; dst translated to MPI_COMM_WORLD rank
 *   coll recorded once per call by communicator rank 0; members = world ranks in comm order
 *        bytes: allreduce/bcast/reduce = count*typesize (full buffer)
 *               allgather/reduce_scatter_block = per-rank chunk
 *               alltoall = sendcount*typesize per peer * comm size
 * Durations are wall time inside the call (MPI_Wtime). For non-blocking calls this is only the
 * posting time — treat Isend durations as meaningless.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static FILE *cage_fp = NULL;
static long cage_step = 0;
static int cage_world_rank = -1;

static void cage_open(void) {
    const char *dir = getenv("CAGE_OUT");
    char path[4096];
    if (!dir) dir = ".";
    mkdir(dir, 0755);
    PMPI_Comm_rank(MPI_COMM_WORLD, &cage_world_rank);
    snprintf(path, sizeof(path), "%s/cage_trace.%d.csv", dir, cage_world_rank);
    cage_fp = fopen(path, "w");
    if (cage_fp) fprintf(cage_fp, "kind,op,src,dst,bytes,step,t0,t1,root,members\n");
}

static int to_world(MPI_Comm comm, int r) {
    MPI_Group g, wg;
    int out = MPI_UNDEFINED;
    if (comm == MPI_COMM_WORLD) return r;
    PMPI_Comm_group(comm, &g);
    PMPI_Comm_group(MPI_COMM_WORLD, &wg);
    PMPI_Group_translate_ranks(g, 1, &r, wg, &out);
    PMPI_Group_free(&g);
    PMPI_Group_free(&wg);
    return out;
}

static long type_bytes(MPI_Datatype t, long count) {
    int sz = 0;
    PMPI_Type_size(t, &sz);
    return (long)sz * count;
}

static void log_p2p(const char *op, int dst_comm_rank, MPI_Comm comm, long bytes,
                    double t0, double t1) {
    if (!cage_fp || dst_comm_rank == MPI_PROC_NULL) return;
    fprintf(cage_fp, "p2p,%s,%d,%d,%ld,%ld,%.9f,%.9f,-1,\n", op, cage_world_rank,
            to_world(comm, dst_comm_rank), bytes, cage_step, t0, t1);
}

static void log_coll(const char *op, MPI_Comm comm, long bytes, int root, double t0, double t1) {
    int crank, csize, i;
    if (!cage_fp) return;
    PMPI_Comm_rank(comm, &crank);
    if (crank != 0) return;
    PMPI_Comm_size(comm, &csize);
    fprintf(cage_fp, "coll,%s,%d,-1,%ld,%ld,%.9f,%.9f,%d,", op, cage_world_rank, bytes, cage_step,
            t0, t1, root >= 0 ? to_world(comm, root) : -1);
    for (i = 0; i < csize; i++) fprintf(cage_fp, i ? ";%d" : "%d", to_world(comm, i));
    fprintf(cage_fp, "\n");
}

/* ------------------------------------------------------------------ lifecycle */
int MPI_Init(int *argc, char ***argv) {
    int rc = PMPI_Init(argc, argv);
    cage_open();
    return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int rc = PMPI_Init_thread(argc, argv, required, provided);
    cage_open();
    return rc;
}

int MPI_Finalize(void) {
    if (cage_fp) { fclose(cage_fp); cage_fp = NULL; }
    return PMPI_Finalize();
}

int MPI_Pcontrol(const int level, ...) {
    if (level == 1) cage_step++;
    return MPI_SUCCESS;
}

/* ------------------------------------------------------------- point-to-point */
int MPI_Send(const void *buf, int count, MPI_Datatype t, int dest, int tag, MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Send(buf, count, t, dest, tag, comm);
    log_p2p("send", dest, comm, type_bytes(t, count), t0, PMPI_Wtime());
    return rc;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype t, int dest, int tag, MPI_Comm comm,
              MPI_Request *req) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Isend(buf, count, t, dest, tag, comm, req);
    log_p2p("isend", dest, comm, type_bytes(t, count), t0, PMPI_Wtime());
    return rc;
}

int MPI_Sendrecv(const void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                 void *rbuf, int rcount, MPI_Datatype rtype, int source, int rtag,
                 MPI_Comm comm, MPI_Status *status) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Sendrecv(sbuf, scount, stype, dest, stag, rbuf, rcount, rtype, source, rtag,
                           comm, status);
    log_p2p("sendrecv", dest, comm, type_bytes(stype, scount), t0, PMPI_Wtime());
    return rc;
}

/* ---------------------------------------------------------------- collectives */
int MPI_Allreduce(const void *s, void *r, int count, MPI_Datatype t, MPI_Op op, MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Allreduce(s, r, count, t, op, comm);
    log_coll("allreduce", comm, type_bytes(t, count), -1, t0, PMPI_Wtime());
    return rc;
}

int MPI_Reduce(const void *s, void *r, int count, MPI_Datatype t, MPI_Op op, int root,
               MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Reduce(s, r, count, t, op, root, comm);
    log_coll("reduce", comm, type_bytes(t, count), root, t0, PMPI_Wtime());
    return rc;
}

int MPI_Bcast(void *buf, int count, MPI_Datatype t, int root, MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Bcast(buf, count, t, root, comm);
    log_coll("broadcast", comm, type_bytes(t, count), root, t0, PMPI_Wtime());
    return rc;
}

int MPI_Allgather(const void *s, int scount, MPI_Datatype st, void *r, int rcount,
                  MPI_Datatype rt, MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Allgather(s, scount, st, r, rcount, rt, comm);
    log_coll("allgather", comm, type_bytes(rt, rcount), -1, t0, PMPI_Wtime());
    return rc;
}

int MPI_Reduce_scatter_block(const void *s, void *r, int rcount, MPI_Datatype t, MPI_Op op,
                             MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int rc = PMPI_Reduce_scatter_block(s, r, rcount, t, op, comm);
    log_coll("reduce_scatter", comm, type_bytes(t, rcount), -1, t0, PMPI_Wtime());
    return rc;
}

int MPI_Alltoall(const void *s, int scount, MPI_Datatype st, void *r, int rcount,
                 MPI_Datatype rt, MPI_Comm comm) {
    int csize;
    double t0 = PMPI_Wtime();
    int rc = PMPI_Alltoall(s, scount, st, r, rcount, rt, comm);
    PMPI_Comm_size(comm, &csize);
    log_coll("alltoall", comm, type_bytes(st, scount) * csize, -1, t0, PMPI_Wtime());
    return rc;
}
