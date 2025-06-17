from pydantic import BaseModel, Field
from typing import Literal, Optional

class CNNConfig(BaseModel):
    kernel_size: int
    filters: int
    stride: int
    dropout_conv: float
    padding: int
    dilation: int
    pool_size: int
    num_layers_conv: int
    smoothing: bool
    TPI_included: bool
    activation: str
    outlayer: str
    smoothfunction: str
    hidden: int
    num_layers: int
    dropout_fc: float

class RaserConfig(BaseModel):
    # General experiment/model fields
    os: str
    id: str
    set: str
    Dataset: str
    Testingset: Optional[str] = None
    arch: Optional[str] = None
    Testing: Optional[int] = None
    Scheduler: Optional[bool] = None
    TPI_included: Optional[bool] = None
    TPI_split: Optional[int] = None
    LR: Optional[float] = None
    WD: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    input_shape: Optional[int] = None
    output_shape: Optional[int] = None
    loss: Optional[Literal['MSE', 'MAE', 'Huber']] = None
    optimizer: Optional[Literal['adam', 'SGD', 'adamw', 'adagrad']] = None
    normalization: Optional[str] = None
    # Image-specific fields
    Image_Size: Optional[int] = None
    input: Optional[Literal['Iradon', 'Sinogram']] = None
    output: Optional[Literal['image', 'Sinogram']] = None
    Ext_Model: Optional[str] = None
    # Data processing specific fields
    sim_rounds: Optional[int] = None
    TPIs: Optional[int] = None
    signal_shape: Optional[int] = None
    angle_count: Optional[int] = None
    img_size: Optional[int] = None
    x2D: Optional[bool] = Field(None, alias='2D')
    # CNN config as a submodel
    cnn: Optional[CNNConfig] = None
