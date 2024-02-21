import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

attrs = ['color', 'shape', 'material', 'size']

def get_target_maps(atoms, atom_grads, attention_maps):
    if atom_grads.size(0) == 1:
        atom_grads = atom_grads.squeeze(0)

    obj_ids = []
    scores = []
    for i, g in enumerate(atom_grads):
        atom = atoms[i]
        if g > 0.8 and 'obj' in str(atom):
            obj_id = str(atom).split(',')[0][-1]
            obj_id = int(obj_id)
            if not obj_id in obj_ids:
                obj_ids.append(obj_id)
                scores.append(g)
    return [attention_maps[id] * scores[i] for i, id in enumerate(obj_ids)]


def valuation_to_attr_string(v, atoms, e, th=0.5):
    """Generate string explanations of the scene.
    """

    st = ''
    for i in range(e):
        st_i = ''
        for j, atom in enumerate(atoms):
            #print(atom, [str(term) for term in atom.terms])
            if 'obj' + str(i) in [str(term) for term in atom.terms] and atom.pred.name in attrs:
                if v[j] > th:
                    prob = np.round(v[j].detach().cpu().numpy(), 2)
                    st_i += str(prob) + ':' + str(atom) + ','
        if st_i != '':
            st_i = st_i[:-1]
            st += st_i + '\n'
    return st


def valuation_to_rel_string(v, atoms, th=0.5):
    l = 15
    st = ''
    n = 0
    for j, atom in enumerate(atoms):
        if v[j] > th and not (atom.pred.name in attrs+['in', '.']):
            prob = np.round(v[j].detach().cpu().numpy(), 2)
            st += str(prob) + ':' + str(atom) + ','
            n += len(str(prob) + ':' + str(atom) + ',')
        if n > l:
            st += '\n'
            n = 0
    return st[:-1] + '\n'


def valuation_to_string(v, atoms, e, th=0.5):
    return valuation_to_attr_string(v, atoms, e, th) + valuation_to_rel_string(v, atoms, th)


def valuations_to_string(V, atoms, e, th=0.5):
    """Generate string explanation of the scenes.
    """
    st = ''
    for i in range(V.size(0)):
        st += 'image ' + str(i) + '\n'
        # for each data in the batch
        st += valuation_to_string(V[i], atoms, e, th)
    return st


def generate_captions(V, atoms, e, th):
    captions = []
    for v in V:
        # for each data in the batch
        captions.append(valuation_to_string(v, atoms, e, th))
    return captions


def save_images_with_captions_and_attention_maps(imgs, attention_maps, captions, folder, img_id, dataset):
    if not os.path.exists(folder):
        os.makedirs(folder)

    figsize = (12, 6)
    # imgs should be denormalized.

    img_size = imgs[0].shape[0]
    attention_maps = np.array([m.cpu().detach().numpy() for m in attention_maps])
    attention_map = np.zeros_like(attention_maps[0])
    for am in attention_maps:
        attention_map += am
    attention_map = attention_map.reshape(32, 32)
    attention_map = cv2.resize(attention_map, (img_size, img_size))
    #attention_map = torch.tensor(attention_map).extend()

    # apply attention maps to filter
    for i, img in enumerate(imgs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2)) #, sharex=True, sharey=True)
        ax1.axis('off')
        ax2.axis('off')

        #e=figsize, dpi=80)
        ax1.imshow(img)
        #ax1.xlabel(captions[i])
        #ax1.tight_layout()
        #ax1.savefig(folder+str(img_id)+'_original.png')

        # ax2.figure(figsize=figsize, dpi=80)
        ax2.imshow(attention_map, cmap='cividis')
        #ax2.set_xlabel(captions[i], fontsize=12)
        #plt.axis('off')
        fig.tight_layout()
        fig.savefig(folder + dataset + '_img' + str(img_id) + '_explanation.svg')

        plt.close()

def save_images_with_captions_and_attention_maps_indivisual(imgs, attention_maps, captions, folder, img_id, dataset):
    if not os.path.exists(folder):
        os.makedirs(folder)

    figsize = (12, 6)
    # imgs should be denormalized.

    img_size = imgs[0].shape[0]
    attention_maps = np.array([m.cpu().detach().numpy() for m in attention_maps])
    attention_map = np.zeros_like(attention_maps[0])
    for am in attention_maps:
        attention_map += am
    attention_map = attention_map.reshape(32,32)
    attention_map = cv2.resize(attention_map, (img_size, img_size))
    #attention_map = torch.tensor(attention_map).extend()

    # apply attention maps to filter
    for i, img in enumerate(imgs):
        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(img)
        plt.xlabel(captions[i])
        plt.tight_layout()
        plt.savefig(folder+str(img_id)+'_original.png')

        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(attention_map, cmap='cividis')
        plt.xlabel(captions[i])
        plt.tight_layout()
        plt.savefig(folder+str(img_id)+'_attention_map.png')

        plt.close()

def denormalize_clevr(imgs):
    """denormalize clevr images
    """
    # normalizing: image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
    return (0.5 * imgs) + 0.5


def denormalize_kandinsky(imgs):
    """denormalize kandinsky images
    """
    return imgs


def to_plot_images_clevr(imgs):
    return [img.permute(1, 2, 0).detach().numpy() for img in denormalize_clevr(imgs)]


def to_plot_images_kandinsky(imgs):
    return [img.permute(1, 2, 0).detach().numpy() for img in denormalize_kandinsky(imgs)]
