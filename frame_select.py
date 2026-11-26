

def get_ref_index_uniform(neighbor_ids, length, config):
    """ select ref frame index uniformly according to num_ref and ref_stride """
    num_ref = config['test_data_loader']['num_ref']
    assert 0 < num_ref <= 10
    if num_ref == 1:
        ref_index = [length//2]
    else:
        ref_index = [i for i in range(0, length + 1, length // (num_ref - 1))]
    if num_ref > 0 and ref_index[-1] >= length - 1:
        ref_index[-1] = length - 1
    assert len(ref_index) == num_ref

    for i in range(len(ref_index)):
        ref_idx = ref_index[i]
        if ref_idx in neighbor_ids:
            top = neighbor_ids[0]
            bottom = neighbor_ids[-1]
            dis_top = ref_idx - top
            dis_bot = bottom - ref_idx
            if top == 0:
                ref_index[i] = bottom + 1
            elif bottom == length - 1:
                ref_index[i] = top - 1
            else:
                ref_index[i] = top - 1 if (dis_top <= dis_bot) else bottom + 1
    return ref_index

