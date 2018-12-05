#include "include/mcm/pass/serializeKeenBay.hpp"


using namespace MVCNN;


void serialize(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb, const char* s){
    std::cout << "> Create Header " << std::endl;
    auto header = getHeader(fg, fbb);

    auto tl = getTaskLists(fg, fbb);

    auto g = CreateGraphFile(*fbb,
        header,
        // 0,
        tl,
        0);

    fbb->Finish(g, "BLOB");  // Serialize the root object

    uint8_t *buf = fbb->GetBufferPointer();
    int size = fbb->GetSize();

    // bool result = flatbuffers::Save(s, *fbb, true);

    std::ofstream myFile;
    myFile.open(s);
    myFile.write((const char*)buf, size );
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MVCNN::TaskList> > >  getTaskLists(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb){

    // auto task = fbb->CreateTask(*fbb,
    //     0,
    //     0,
    //     SpecificTask_DPUTask,
    //     0
    // );
    // task.add_task_lists

    (void)fg;

    DPUInvariantFieldsBuilder dpu_iv_B(*fbb);
    // dpu_iv_B.add_inputs_are_sparse(fg->tasks[0].sparse_in);
    auto dpu_iv_task = dpu_iv_B.Finish();

    DPUTaskBuilder dpuTB(*fbb);
    dpuTB.add_invariant(dpu_iv_task);
    auto dputask = dpuTB.Finish();


    TaskBuilder tb(*fbb);
    tb.add_task_type(SpecificTask_DPUTask);
    tb.add_task(dputask.Union());
    auto task = tb.Finish();

    std::vector<flatbuffers::Offset<Task>> v = {task};

    auto vv = fbb->CreateVector(v);

    TaskListBuilder tasklist(*fbb);
    tasklist.add_content(vv);
    auto tl = tasklist.Finish();

    std::vector<flatbuffers::Offset<MVCNN::TaskList>> v2 = {tl};

    auto tls = fbb->CreateVector(v2);

    return tls;
}


flatbuffers::Offset<SummaryHeader> getHeader(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb){
    auto s = fbb->CreateString(fg->githash);
    (void)s;
    auto v = CreateVersion(*fbb, fg->version[0], fg->version[1], fg->version[2], s);
    // auto v = Version(3,0,1);

    auto r = CreateResources(*fbb,
        fg->shaveMask,
        fg->nce1Mask,
        fg->dpuMask,
        fg->leonCMX,
        fg->nnCMX,
        fg->ddrScratch
    );

    auto d = CreateIndirectDataReference(*fbb, 0, 0);

    auto t = CreateTensorReference(*fbb,
        fbb->CreateVector(fg->dims, 4),
        fbb->CreateVector(fg->strides, 5),
        0,
        0,
        d,
        MemoryLocation_NULL
    );

    flatbuffers::Offset<MVCNN::TensorReference> tt[] = {t, t};
    auto tv = fbb->CreateVector(tt, 2);

    auto src = CreateSourceStructure(*fbb,0,0 );

    auto h = CreateSummaryHeader(*fbb,
        v,
        tv,
        tv,
        fg->taskAmount,
        fg->layerAmount,
        r,
        src
    );
    return h;
}