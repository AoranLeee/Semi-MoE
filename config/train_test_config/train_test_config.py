import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi
import torch
import os
from PIL import Image

def print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, 
                        train_loss_unsup, train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_sup3 = train_loss_sup_3 / num_batches['train_sup']
    train_epoch_loss_unsup = train_loss_unsup / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    
    print('-' * print_num)
    print('| Train Sup Loss Seg: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss Sdf: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss Bnd: {:.4f}'.format(train_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_unsup).ljust(print_num_minus, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')

    print('-' * print_num)
    
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss

def print_val_loss(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']
    val_epoch_loss_sup3 = val_loss_sup_3 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss Seg: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Val Sup Loss Sdf: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Val Sup Loss Bnd: {:.4f}'.format(val_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3


def print_train_loss_sup(train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss

def print_val_loss_sup(val_loss, num_batches, print_num, print_num_minus):
    val_epoch_loss = val_loss / num_batches['val']
    print('-' * print_num)
    print('| Val Loss: {:.4f}'.format(val_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss


def print_train_eval_sup(num_classes, score_list_train, mask_list_train, print_num):

    if num_classes == 2:
        eval_list = evaluate(score_list_train, mask_list_train)
        print('| Train Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    else:
        eval_list = evaluate_multi(score_list_train, mask_list_train)

        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc: {}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Train mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    return eval_list, train_m_jc


def print_val_eval_sup(num_classes, score_list_val, mask_list_val, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_val, mask_list_val)
        print('| Val Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    else:
        eval_list = evaluate_multi(score_list_val, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Val mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    return eval_list, val_m_jc

def print_val_eval(num_classes, score_list_val1, score_list_val2, mask_list_val, print_num):
    if num_classes == 2:
        eval_list1 = evaluate(score_list_val1, mask_list_val)
        eval_list2 = evaluate(score_list_val2, mask_list_val)
        print('| Val Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    else:
        eval_list1 = evaluate_multi(score_list_val1, mask_list_val)
        eval_list2 = evaluate_multi(score_list_val2, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc 1: {}  '.format(eval_list1[0]).ljust(print_num, ' '), '| Val  Jc 2: {}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc 1: {}  '.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Val mJc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val mJc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val mDc 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Val mDc 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    return eval_list1, eval_list2, val_m_jc1, val_m_jc2

def save_val_best_sup_2d(num_classes, best_list, models, score_list_val, name_list_val, eval_list, path_trained_model, path_seg_results, palette, model_name):
    if best_list[1] >= eval_list[1]:  
        return best_list

    best_list = eval_list  
    checkpoint_path = os.path.join(path_trained_model, f'best_{model_name}_Jc_{best_list[1]:.4f}.pth')
    torch.save({name: model.state_dict() for name, model in models.items()}, checkpoint_path)

    if num_classes == 2:
        score_list_val = torch.softmax(score_list_val, dim=1)
        pred_results = (score_list_val[:, 1, :, :].cpu().numpy() > eval_list[0]).astype(np.uint8)
    else:
        pred_results = torch.argmax(score_list_val, dim=1).cpu().numpy()

    assert len(name_list_val) == pred_results.shape[0]
    for i in range(len(name_list_val)):
        color_results = Image.fromarray(pred_results[i], mode='P')
        color_results.putpalette(palette)
        color_results.save(os.path.join(path_seg_results, name_list_val[i]))

    return best_list


def save_val_best_2d(num_classes, best_model, best_list, best_result, model1, model2, score_list_val_1, score_list_val_2, name_list_val, eval_list_1, eval_list_2, path_trained_model, path_seg_results, palette):

    if eval_list_1[1] < eval_list_2[1]:
        if best_list[1] < eval_list_2[1]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'

            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result2', best_list[1])))

            if num_classes == 2:
                score_list_val_2 = torch.softmax(score_list_val_2, dim=1)
                pred_results = score_list_val_2[:, 1, ...].cpu().numpy()
                pred_results[pred_results > eval_list_2[0]] = 1
                pred_results[pred_results <= eval_list_2[0]] = 0
            else:
                pred_results = torch.max(score_list_val_2, 1)[1]
                pred_results = pred_results.cpu().numpy()

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[1] < eval_list_1[1]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'

            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result1', best_list[1])))

            if num_classes == 2:
                score_list_val_1 = torch.softmax(score_list_val_1, dim=1)
                pred_results = score_list_val_1[:, 1, ...].cpu().numpy()
                pred_results[pred_results > eval_list_1[0]] = 1
                pred_results[pred_results <= eval_list_1[0]] = 0
            else:
                pred_results = torch.max(score_list_val_1, 1)[1]
                pred_results = pred_results.cpu().numpy()

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result


    return best_list, best_model, best_result


def print_best_sup(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')


def print_test_eval(num_classes, score_list_test, mask_list_test, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_test, mask_list_test)
        print('| Test Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
    else:
        eval_list = evaluate_multi(score_list_test, mask_list_test)
        np.set_printoptions(precision=4, suppress=True)
        print('| Test  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Test mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')

    return eval_list


def save_test_2d(num_classes, score_list_test, name_list_test, threshold, path_seg_results, palette):

    if num_classes == 2:
        score_list_test = torch.softmax(score_list_test, dim=1)
        pred_results = score_list_test[:, 1, ...].cpu().numpy()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

    else:
        pred_results = torch.max(score_list_test, 1)[1]
        pred_results = pred_results.cpu().numpy()

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

