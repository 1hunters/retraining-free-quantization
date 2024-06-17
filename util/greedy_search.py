from .utils import model_profiling, reset_batchnorm_stats
import torch
import torch.distributed as dist
import math
from .qat import get_quantized_layers
from .mpq import switch_bit_width

def dist_all_reduce_tensor(tensor):
    """ Reduce to all ranks """
    world_size = dist.get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def forward_loss(
        model, criterion, input, target, meter, soft_target=None,
        soft_criterion=None, return_soft_target=False, return_acc=False, eval_mode=False):
    
    """forward model and return loss"""
    if eval_mode:
        model.eval()
    topk = (1, 5)
    output = model(input)

    loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in topk:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # allreduce
    tensor = dist_all_reduce_tensor(tensor)
    # cache to meter
    tensor = tensor.cpu().detach().numpy()
    bs = (tensor.size-1)//2
    for i, k in enumerate(topk):
        error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
        if return_acc and k == 1:
            top1_error = sum(error_list) / len(error_list)
            return loss, top1_error
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(tensor[0])
    if return_soft_target:
        return loss, torch.nn.functional.softmax(output, dim=1)
    return loss



def adjust_one_layer_bit_width(layer, bn, next_bits:int, reduce: bool, tensor: str):
    if not reduce:
        if tensor == 'weight':
            layer.bits = (next_bits, layer.bits[1])
            bn.switch_bn(layer.bits)
            return layer.bits[0]
        else:
            layer.bits = (layer.bits[0], next_bits)
            bn.switch_bn(layer.bits)
            return layer.bits[1]
    else:
        if tensor == 'weight':
            layer.bits = (next_bits, layer.bits[1])
            bn.switch_bn(layer.bits)
            return layer.bits[0]
        else:
            layer.bits = (layer.bits[0], next_bits)
            bn.switch_bn(layer.bits)
            return layer.bits[1]

def get_layer_wise_conf(layers, tensor):

    return [l.bits[0] if tensor=='weight' else l.bits[1] for l in layers]  


def reset_bit_cands(model: torch.nn.Module, reset=True):
    from quan.func import QuanConv2d
    
    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d):
            if reset:
                print(name, 'bits_cands to', module.reset_bits_cands())
            else:
                print(name, module.weight_bit_cands, module.act_bit_cands)


def search(loader, model, criterion, metrics, cfgs, epoch=0, start_bits=6, init_w=None, init_a=None):
    constraint_type, target_size = metrics
    quan_scheduler = cfgs.quan
    target_bits = cfgs.target_bits

    if init_a and init_w:
        from .qat import set_bit_width
        set_bit_width(model, init_w, init_a)
        print("init from", init_w, init_a)
    assert constraint_type in ['model_size', 'bitops']

    loader.sampler.set_epoch(epoch)
    iterator = iter(loader)

    data_for_bn_rest = [next(iterator) for _ in range(3)]

    model.eval()
    reset_batchnorm_stats(model)
    
    quantized_layers, bn = get_quantized_layers(model)
    target_size: list

    lut, lut_complexity = [], []
    for _ in range(2):
        for layer in quantized_layers:
            lut.append(0)
            
            lut_complexity.append(0)

    print("searching...")
    configs = []
    model.eval()
    
    bitops, model_size = model_profiling(model)
    start_complexity = bitops if constraint_type == 'bitops' else model_size

    smallest_bit_width = 3 if max(target_size) >= 5.0 else 2
    # smallest_bit_width = 2 
    model.train()

    print('smallest bits', smallest_bit_width)

    def bops_map_to_bits(bops, arch='resnet18'):
        if 'mobilenetv2' in arch:
            
            if 5.0 <= bops <= 5.8:
                return 4
            
            if 3.3 <= bops <= 3.8:
                return 3
            
            return 4
        elif 'efficientnet' in arch:
            if 6.3 <= bops <= 7.1:
                return 4
            
            if 3.3 <= bops <= 4.5:
                return 3
            
            return 3
        elif 'resnet18' in arch:
            if 31 <= bops <= 36:
                return 4
            
            if 20 < bops <= 23.9:
                return 3
            
            if bops <= 20:
                return 2

    done_w, done_a = False, False
    w_init, a_init = False, False
    while True:
        input, target = next(iterator)
        input, target = input.cuda(), target.cuda()

        print(f"current bitops {round(bitops, 2)}")
        metric = bitops if constraint_type == 'bitops' else model_size

        acc_scale = 1.5

        # sc = 1.1 # for W2A3-ResNet18
        sc = .95
        
        if not done_w and not w_init:
            bits = bops_map_to_bits(max(target_size), cfgs.arch)
            print(bitops, model_size)

            switch_bit_width(model, quan_scheduler, wbit=bits, abits=start_bits)
            w_init = True
            w_target_bitops, _ = model_profiling(model) # only support bops now
            switch_bit_width(model, quan_scheduler, wbit=start_bits, abits=start_bits)
            w_target_bitops *= sc
            print(w_target_bitops)

            if init_a and init_w:
                set_bit_width(model, init_w, init_a)
            # metric = bitops if constraint_type == 'bitops' else model_size

        if metric <= w_target_bitops and not done_w:
            done_w = True
            lut = [0 for _ in range(len(quantized_layers) * 2)]
            lut_complexity = [0 for _ in range(len(quantized_layers) * 2)]

            print('weight done...')
        
        if done_w and not done_a and metric < max(target_size):
            done_a = True
        
        # if metric < max(target_size) and (done_w and done_a):
        if metric < max(target_size) :
            configs.append((max(target_size), get_layer_wise_conf(quantized_layers, tensor='weight'), get_layer_wise_conf(quantized_layers, tensor='act')))
            target_size.remove(max(target_size))
            done_w, done_a, w_init = False, False, False
        
        if len(target_size) == 0:
            break

        for idx, layer in enumerate(quantized_layers):
            wbits, abits = layer.bits

            for mode in ['+', '-']:
                
                if mode == '+':
                    overall_idx  = idx * 2 + 1
                    # continue
                else:
                    overall_idx  = idx * 2 
                lut[overall_idx] = 0.
                lut_complexity[overall_idx] = 0.

                if mode == '-':
                    if (wbits <= smallest_bit_width and not done_w):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue
                else:
                    if (wbits >= max(target_bits) and not done_w):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue

                if mode == '-':
                    if abits <= smallest_bit_width and (done_w and not done_a):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue
                else:
                    if abits >= max(target_bits) and (done_w and not done_a):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue

                if mode == '-':
                    if wbits > smallest_bit_width and not done_w:
                        next_wbits_index = target_bits.index(wbits) + 1

                        if wbits == min(layer.current_bit_cands_w):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_wbits = target_bits[next_wbits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_wbits, reduce=False, tensor='weight')
                    
                    if abits > smallest_bit_width and (done_w and not done_a):
                        next_abits_index = target_bits.index(abits) + 1

                        if abits == min(layer.current_bit_cands_a):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_abits = target_bits[next_abits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_abits, reduce=False, tensor='act')
                else:
                    if wbits < max(target_bits) and not done_w:
                        next_wbits_index = target_bits.index(wbits) - 1

                        if wbits == max(target_bits):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_wbits = target_bits[next_wbits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_wbits, reduce=False, tensor='weight')
                    
                    if abits < max(target_bits) and (done_w and not done_a):
                        next_abits_index = target_bits.index(abits) - 1

                        if abits == max(target_bits):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_abits = target_bits[next_abits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_abits, reduce=False, tensor='act')
            
                with torch.no_grad():
                    # calibrate_batchnorm_state(model, loader, 15, reset=True, distributed_training=False)

                    # calibrate_batchnorm_state(model, loader=data_for_bn_rest, reset=True, distributed_training=True)
                    _, top1_error = forward_loss(model, criterion, input, target, None, return_acc=True, eval_mode=False)
                # lut[overall_idx] += top1_error
                # lut[overall_idx] += (top1_error - start_top1_error)
                lut[overall_idx] = top1_error

                tmp_bitops, tmp_model_size = model_profiling(model)
                comp = tmp_bitops if constraint_type == 'bitops' else tmp_model_size
                # lut_complexity[overall_idx] += (start_complexity - comp)
                lut_complexity[overall_idx] += -comp

                if mode == '-':
                    if wbits > smallest_bit_width and not done_w:
                        adjust_one_layer_bit_width(layer, bn[idx], wbits, reduce=True, tensor='weight')

                    if abits > smallest_bit_width and (done_w and not done_a):
                        adjust_one_layer_bit_width(layer, bn[idx], abits, reduce=True, tensor='act')
                else:
                    if wbits >= smallest_bit_width and not done_w:
                        adjust_one_layer_bit_width(layer, bn[idx], wbits, reduce=True, tensor='weight')

                    if abits >= smallest_bit_width and (done_w and not done_a):
                        adjust_one_layer_bit_width(layer, bn[idx], abits, reduce=True, tensor='act')
    
        # if wbits > smallest_bit_width:
        #     adjust_one_layer_bit_width(layerw, bn[idxw], wbits, reduce=True, tensor='weight')
            # print(f"top-1 error {top1_error}")
        
        tmp_lut = []
        max_acc, max_comp = 0, 0
        min_acc, min_comp = 0, 0
        for acc, comp in zip(lut, lut_complexity):
            max_acc = acc if (acc > max_acc and acc is not math.inf) else max_acc
            min_acc = acc if (acc < min_acc and acc is not math.inf) else min_acc

            max_comp = comp if (comp > max_comp and comp is not math.inf) else max_comp
            min_comp = comp if (comp < min_comp and comp is not math.inf) else min_comp
        
        for acc, comp in zip(lut, lut_complexity):
            if acc == math.inf:
                tmp_lut.append(math.inf)
                continue

            tmp_lut.append(acc_scale*((acc-min_acc)/(max_acc-min_acc)) - (comp-min_comp)/(max_comp-min_comp))

        best_idx = tmp_lut.index(min(tmp_lut))
        print("current optim metric", min(tmp_lut), 'min top-1 error', lut[best_idx])
        
        assert best_idx is not math.inf

        if not done_w:
            best_layer_index_w = best_idx // 2
            best_layer_wbits = quantized_layers[best_layer_index_w].bits[0]
            offset = 1 if best_idx % 2 == 0 else -1

            if best_layer_wbits > smallest_bit_width:
                
                next_w_bit_index = target_bits.index(best_layer_wbits) + offset
                next_w_bit = target_bits[next_w_bit_index]
                wnew_bit_width = adjust_one_layer_bit_width(quantized_layers[best_layer_index_w], bn[best_layer_index_w], next_w_bit, reduce=False, tensor='weight')
                print(f"layer {best_layer_index_w} weight: bit-width {best_layer_wbits} -> {wnew_bit_width}")
            else:
                print(f"layer {best_layer_index_w} weight bit-width {best_layer_wbits} not change")
        
        if done_w and not done_a:
            best_layer_index_a = best_idx // 2
            best_layer_abits = quantized_layers[best_layer_index_a].bits[1]
            offset = 1 if best_idx % 2 == 0 else -1

            if best_layer_abits > smallest_bit_width:
                next_a_bit_index = target_bits.index(best_layer_abits) + offset
                next_a_bit = target_bits[next_a_bit_index]
                anew_bit_width = adjust_one_layer_bit_width(quantized_layers[best_layer_index_a], bn[best_layer_index_a], next_a_bit, reduce=False, tensor='act')
                print(f"layer {best_layer_index_a} act: bit-width {best_layer_abits} -> {anew_bit_width}")
            else:
                print(f"layer {best_layer_index_a} act bit-width {best_layer_abits} not change")
        
        print("")
        print("weight bit-width assignment", get_layer_wise_conf(quantized_layers, tensor='weight'))
        print("activs bit-width assignment", get_layer_wise_conf(quantized_layers, tensor='act'))
        print('-'*50)

        epoch += 1
        
        bitops, model_size = model_profiling(model)

    return configs
        

