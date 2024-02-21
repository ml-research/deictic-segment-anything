import torch
import torch.nn as nn

from .neural_utils import MLP, LogisticRegression


################################
# Valuation functions for YOLO #
################################

class YOLOColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(YOLOColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 4:7]
        return (a * z_color).sum(dim=1)


class YOLOShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(YOLOShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 7:10]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class YOLOInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(YOLOInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        return z[:, -2]


class YOLOClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(YOLOClosebyValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)
        dist = torch.norm(c_1 - c_2, dim=0).unsqueeze(-1)
        return self.logi(dist).squeeze()

    def to_center(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        y = (z[:, 1] + z[:, 3]) / 2
        return torch.stack((x, y))


class YOLOOnlineValuationFunction(nn.Module):
    """The function v_online.
    """

    def __init__(self, device):
        super(YOLOOnlineValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, z_3, z_4, z_5):
        """The function to compute the probability of the online predicate.

        The closed form of the linear regression is computed.
        The error value is fed into the 1-d logistic regression function.

        Args:
            z_i (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        X = torch.stack([self.to_center_x(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        Y = torch.stack([self.to_center_y(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        # add bias term
        X = torch.cat([torch.ones_like(X), X], dim=2)
        X_T = torch.transpose(X, 1, 2)
        # the optimal weights from the closed form solution
        W = torch.matmul(torch.matmul(
            torch.inverse(torch.matmul(X_T, X)), X_T), Y)
        diff = torch.norm(Y - torch.sum(torch.transpose(W, 1, 2)
                                        * X, dim=2).unsqueeze(-1), dim=1)
        self.diff = diff
        return self.logi(diff).squeeze()

    def to_center_x(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        return x

    def to_center_y(self, z):
        y = (z[:, 1] + z[:, 3]) / 2
        return y


##########################################
# Valuation functions for slot attention #
##########################################


class SlotAttentionInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(SlotAttentionInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (tensor): A one-hot tensor of image e.g. [img1, img2]
            #### x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        ##return z[:, 0]
        objectness = z[:, 0]
        image_ids = z[:, -1]
        mask = x == image_ids
        return objectness * mask.float()
        # return (z * x).sum(dim=1)[:, 0]


class SlotAttentionShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self, device):
        super(SlotAttentionShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 4:7]
        return (a * z_shape).sum(dim=1)


class SlotAttentionSizeValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(SlotAttentionSizeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 7:9]
        return (a * z_size).sum(dim=1)


class SlotAttentionMaterialValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(SlotAttentionMaterialValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 9:11]
        return (a * z_material).sum(dim=1)


class SlotAttentionColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:19]
        return (a * z_color).sum(dim=1)

class SlotAttentionLessColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionLessColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, gray, red, yellow]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:15]
        # z_color = z[:, 10:14]
        return (a * z_color).sum(dim=1)

class SlotAttentionColor1ValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionColor1ValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:19]
        return (a * z_color).sum(dim=1)

class SlotAttentionRightSideValuationFunction(nn.Module):
    """The function v_rightside.
    """

    def __init__(self, device):
        super(SlotAttentionRightSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionLeftSideValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionLeftSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness

class SlotAttentionMoreThanTwoObjectsValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionMoreThanTwoObjectsValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        objectness = z.max(dim=2) # (B, n_objects, )
        num_objects = objectness.sum(dim=1) # (B, )

        diff = num_objects - torch.tensor(2.0).to(self.device)
        prob = self.logi(diff).squeeze()  # (B, )
        return prob

class SlotAttentionFrontValuationFunction(nn.Module):
    """The function v_infront.
    """

    def __init__(self, device):
        super(SlotAttentionFrontValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=6, output_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        xyz_1 = z_1[:, 1:4]
        xyz_2 = z_2[:, 1:4]
        xyzxyz = torch.cat([xyz_1, xyz_2], dim=1)
        prob = self.logi(xyzxyz).squeeze()  # (B,)
        objectness = z_1[:, 0] * z_2[:, 0]  # (B,)
        return prob * objectness


class SlotAttentionLeftOfValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionLeftOfValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (tensor): one-hot tensor of input images.
        Returns:
            A batch of probabilities.
        """

        """ original differentiable implementation
        x_1 = z_1[:, 1]  # (B, )
        x_2 = z_2[:, 1]  # (B, )
        # dist = torch.norm(x_1 - x_2)
        diff = x_1 - x_2
        prob = self.logi(diff.unsqueeze(-1)).squeeze()  # (B, )
        objectness = z_1[:, 0] * z_2[:, 0]  # (B, )

        id_1 = z_1[:, -1]
        id_2 = z_2[:, -1]

        mask = (id_1 == id_2).float()
        return prob * objectness * mask
        """
        x_1 = z_1[:, 1]  # (B, )
        x_2 = z_2[:, 1]  # (B, )
        return ((x_1 - x_2) < 0).float()


class SlotAttentionCountValuationFunction(nn.Module):
    """The function v_count(img, N).
    """

    def __init__(self, device):
        super(SlotAttentionCountValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)
        self.beta = 1e-2

    def forward(self, n, zs_list):
        """
        Args:
            zs_list (tensor): 2-d tensor (B * E * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (tensor): batch of image ids.

        Returns:
            A batch of probabilities.
        """
        objectness = zs_list[:, :, 0]
        sum_objectness = objectness.sum(dim=1).unsqueeze(-1)
        diff = torch.abs(n - sum_objectness)
        return self.logi(diff).squeeze()


class SlotAttentionQuery3ValuationFunction(nn.Module):
    """The function v_query3(query, Querytype, QueryColor, QueryPosition).
    """

    def __init__(self, device):
        super(SlotAttentionQuery3ValuationFunction, self).__init__()
        self.device = device

    def forward(self, query, a_query_type, a_query_color, a_query_position):
        """
        Args:
            query (tensor): tensor (B * D), the tensor-based query representation.
                [delete, apend, reverse, sort, \
                    cyan, gray, red, yellow,\
                    1st, 2nd, 3rd]
            a_query_type (tensor): one-hot encoding of the query type.
            a_query_color (tensor): one-hot encoding of the query color.
            a_query_position (tensor): one-hot encoding of the query position.

        Returns:
            A batch of probabilities.
        """
        z_query_type = query[:, 0:4]
        z_query_color = query[:, 4:8]
        z_query_position = query[:, 8:11]
        prob =  (a_query_type * z_query_type).sum(dim=1) * \
            (a_query_color * z_query_color).sum(dim=1) * \
                (a_query_position * z_query_position).sum(dim=1)
        return prob


class SlotAttentionQuery2ValuationFunction(nn.Module):
    """The function v_query3(query, Querytype, QueryColor, QueryPosition).
    """
    def __init__(self, device):
        super(SlotAttentionQuery2ValuationFunction, self).__init__()
        self.device = device

    def forward(self, query, a_query_type, a_query_position):
        """
        Args:
            query (tensor): tensor (B * D), the tensor-based query representation.
                [delete, apend, reverse, sort, \
                    cyan, blue, yellow, purple, red, green, gray, brown\
                    1st, 2nd, 3rd]
            a_query_type (tensor): one-hot encoding of the query type.
            a_query_position (tensor): one-hot encoding of the query position.

        Returns:
            A batch of probabilities.
        """
        z_query_type = query[:, 0:4]
        z_query_position = query[:, 8:11]
        prob =  (a_query_type * z_query_type).sum(dim=1) * \
            (a_query_position * z_query_position).sum(dim=1)
        return prob