import math
import os

import torch
from safetensors import safe_open


# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}


class Checkpoint:
    """
    Initialize a Checkpoint object from a directory containing .safetensors files.
    
    Args:
        path (str): The directory path containing .safetensors files.
        device (torch.device): The device (e.g., 'cuda:0', 'cpu') on which to load tensors.
    
    Attributes:
        device_str (str): The device string representation (e.g., 'cuda:0' or 'cpu').
        tensor_name_to_file (dict): A mapping from tensor name to the file path where it is stored.
    """

    def __init__(self, path: str, device: torch.device):
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str

        # Read from all files ending with .safetensors in the checkpoint directory
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        # Build a mapping from tensor name to (file, key)
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        """
        Retrieve a tensor by name, handling both MoE weights (in block-based MXFP4 format) and other weights/biases.
        
        Args:
            name (str): The name of the tensor to retrieve. This name is used to look up the actual tensor name in the PARAM_NAME_MAP.
        
        Returns:
            torch.Tensor: The requested tensor. For MoE weights, it is returned in MXFP4 format with bfloat16 dtype; for other weights/biases, it is returned as a standard tensor.
        
        Note:
            - If the name is found in PARAM_NAME_MAP, it is matched against a tuple of (blocks_name, scales_name) for MoE weights.
            - If the name is not found in PARAM_NAME_MAP, it is treated as a direct tensor name.
            - The method uses a pattern-matching approach to distinguish between MoE weights and other tensors.
        """

        match PARAM_NAME_MAP.get(name, name):
            case (blocks_name, scales_name):
                # MoE weights: are in block-based MXFP4 format
                return self._get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
            case tensor_name:
                # MoE biases and other weights
                return self._get_tensor(tensor_name)

    def _get_tensor(self, name: str) -> str:
        """
        Retrieve a tensor from a checkpoint file by name.
        
        Args:
            name (str): The name of the tensor to retrieve from the checkpoint.
        
        Returns:
            str: The tensor data as a string representation.
        
        Raises:
            AssertionError: If the tensor name is not found in the tensor name to file mapping.
            Exception: If there is an error opening or reading the tensor file (handled by safe_open).
        """

        assert name in self.tensor_name_to_file, f"Tensor {name} not found in checkpoint."
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        """
        Internal function to convert FP4 blocks and scales tensors into a full tensor representation using lookup tables and exponentiation.
        
        Args:
            blocks_name (str): Name of the tensor containing the FP4 block data in the checkpoint.
            scales_name (str): Name of the tensor containing the scaling factors (offset from 127) in the checkpoint.
            dtype (torch.dtype, optional): Data type for the output tensor. Defaults to torch.bfloat16.
            rows_per_chunk (int, optional): Number of rows to process in each chunk to optimize memory and performance. Defaults to 16384 * 516.
        
        Returns:
            torch.Tensor: A reshaped tensor of shape (*prefix_shape, G * B * 2) where each FP4 value is expanded into two 4-bit values (low and high nibbles) with their corresponding exponent applied.
        
        Raises:
            AssertionError: If either blocks_name or scales_name is not found in the checkpoint's tensor mapping, or if the shapes of blocks and scales do not match.
        """

        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total   = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    def _get_mxfp4_tensor_copy(self, blocks_name: str, scales_name: str, dtype: torch.dtype = torch.bfloat16):
        """
        Internal helper method to convert MXFP4 quantized tensor blocks and scales into a floating-point tensor of specified dtype.
        
        Args:
            blocks_name: Name of the tensor storing the MXFP4 block data (low and high nibbles).
            scales_name: Name of the tensor storing the quantization scales.
            dtype: Target data type for the output tensor (default is torch.bfloat16).
        
        Returns:
            A tensor of the specified dtype, where each element represents the decoded floating-point value
            from the MXFP4 quantized format using the provided blocks and scales.
        
        Note:
            This method performs the following steps:
            - Loads the MXFP4 block data and splits each 8-bit value into low (nibbles 0-3) and high (nibbles 4-7) nibbles.
            - Interleaves the low and high nibbles to form a 16-bit value.
            - Loads the scale values, converts them to int32, and subtracts a bias of 127.
            - Uses a lookup table (FP4_VALUES) to map the MXFP4 block values to floating-point values.
            - Applies the scale via ldexp (multiply by 2^scale) to reconstruct the original floating-point values.
            - Reshapes the output tensor to match the original shape, excluding the last two dimensions.
        """

        loaded_blocks = self._get_tensor(blocks_name)
        # Split it into low and high nibbles, upcast to bytes, and interleave (for swiglu)
        loaded_blocks_lo = loaded_blocks & 0x0F
        loaded_blocks_hi = loaded_blocks >> 4
        loaded_blocks = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
        loaded_blocks = loaded_blocks.view(*loaded_blocks.shape[:-2], loaded_blocks.shape[-2] * 2)

        loaded_scales = self._get_tensor(scales_name)
        # Upcast to int32 and subtract bias
        loaded_scales = loaded_scales.int() - 127

        # Convert MXFP4 numbers into target dtype
        fp4_values = torch.tensor(FP4_VALUES, dtype=dtype, device=self.device_str)
        loaded_tensor = torch.ldexp(fp4_values[loaded_blocks.int()], loaded_scales.unsqueeze(-1))
        loaded_tensor = loaded_tensor.view(*loaded_tensor.shape[:-2], -1)
        return loaded_tensor
