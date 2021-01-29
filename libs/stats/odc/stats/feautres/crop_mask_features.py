"""
crop mask feautres for AI
"""
from typing import Optional, Tuple

import xarray as xr
from odc.algo import erase_bad, geomedian_with_mads, to_rgba
from odc.algo.io import load_enum_filtered
from odc.algo.io import load_with_native_transform
from odc.stats.model import Task
from xarray import DataArray

from .model import OutputProduct, StatsPluginInterface
from .. import _plugins


class CropMaskFeautures(StatsPluginInterface):
    """
    build crop mask freatures from geomedian_with_mads
    product is s2_l2a
    """

    sr_max = 10000
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    # todo add chiprs_uri
    chirps_uri = "s3-to-tbd"

    def __init__(
        self,
        resampling: str = "bilinear",
        bands: Optional[Tuple[str, ...]] = None,
        filters: Optional[Tuple[int, int]] = (2, 5),
        work_chunks: Tuple[int, int] = (400, 400),
        mask_band: str = "SCL",
        cloud_classes=(
            "cloud shadows",
            "cloud medium probability",
            "cloud high probability",
            "thin cirrus",
        ),
        basis_band=None,
        aux_names=dict(smad="SMAD", emad="EMAD", bcmad="BCMAD", count="COUNT"),
        rgb_bands=None,
        rgb_clamp=(1, 3_000),
    ):
        # TODO: mapping to red, green, ... , the meaningful aliases
        if bands is None:
            bands = (
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            )
            if rgb_bands is None:
                rgb_bands = ("B04", "B03", "B02")

        # Dictionary mapping full data names to simpler alias names
        self.bandnames_dict = {
            # "nir_1": "nir",
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B05": "red_edge_1",
            "B06": "red_edge_2",
            "B07": "red_edge_3",
            "B08": "nir",
            "B08A": "nir_narrow",
            "B11": "swir_1",
            "B12": "swir_2",
        }

        self.resampling = resampling
        self.bands = tuple(bands)
        self._basis_band = basis_band or self.bands[0]
        self._renames = aux_names
        self.rgb_bands = rgb_bands
        self.rgb_clamp = rgb_clamp
        self.aux_bands = tuple(
            self._renames.get(k, k) for k in ("smad", "emad", "bcmad", "count")
        )

        # collect from calculate_indices
        self.calculated_bands = ("NDVI", "LAI", "MNDWI")
        # slop band
        self.slope_band = ("slope",)
        # chirps rainfall band
        self.rainfall_band = ("rainfall",)

        self._mask_band = mask_band
        self.filters = filters
        self.cloud_classes = tuple(cloud_classes)
        self._work_chunks = work_chunks

    def bands_to_rename(self, ds: xr.Dataset) -> xr.Dataset:
        """
        rename the variables according to the band_dict
        """
        return {a: b for a, b in self.bandnames_dict.items() if a in ds.variables}

    def product(self, location: Optional[str] = None, **kw) -> OutputProduct:
        name = "crop_mask_features"
        short_name = "cm_feat"
        version = "0.0.1"

        if location is None:
            bucket = "crop-mask-dev"
            location = f"s3://{bucket}/{name}/v{version}"
        else:
            location = location.rstrip("/")

        # add additional bands for products
        measurements = (
            self.bands
            + self.aux_bands
            + self.calculated_bands
            + self.slope_band
            + self.rainfall_band
        )

        properties = {
            "odc:file_format": "GeoTIFF",
            "odc:producer": "ga.gov.au",
            "odc:product_family": "statistics",  # TODO: ???
            "platform": "sentinel-2",
        }

        return OutputProduct(
            name=name,
            version=version,
            short_name=short_name,
            location=location,
            properties=properties,
            measurements=measurements,
            href=f"https://collections.digitalearth.africa/product/{name}",
        )

    def input_data(self, task: Task) -> xr.Dataset:
        basis = self._basis_band
        chunks = {"y": -1, "x": -1}
        groupby = "solar_day"

        erased = load_enum_filtered(
            task.datasets,
            self._mask_band,
            task.geobox,
            categories=self.cloud_classes,
            filters=self.filters,
            groupby=groupby,
            resampling=self.resampling,
            chunks={},
        )

        xx = load_with_native_transform(
            task.datasets,
            self.bands,
            task.geobox,
            lambda xx: xx,
            groupby=groupby,
            basis=basis,
            resampling=self.resampling,
            chunks=chunks,
        )

        xx = erase_bad(xx, erased)
        return xx

    @staticmethod
    def add_indices(ds: xr.Dataset) -> xr.Dataset:
        """
        calculate_indices was done here
        """
        index_dict = {
            # Normalised Difference Vegation Index, Rouse 1973
            "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
            # Leaf Area Index, Boegh 2002
            "LAI": lambda ds: (
                3.618
                * (
                    (2.5 * (ds.nir - ds.red))
                    / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)
                )
                - 0.118
            ),
            # Modified Normalised Difference Water Index, Xu 2006
            "MNDWI": lambda ds: (ds.green - ds.swir_1) / (ds.green + ds.swir_1),
        }
        for key, func in index_dict.items():
            index_array = func(ds)
        return ds

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        scale = 1 / 10_000
        cfg = dict(
            maxiters=1000,
            num_threads=1,
            scale=scale,
            offset=-1 * scale,
            reshape_strategy="mem",
            out_chunks=(-1, -1, -1),
            work_chunks=self._work_chunks,
            compute_count=True,
            compute_mads=True,
        )

        gm = geomedian_with_mads(xx, **cfg)
        # self._renames.update(self.bandnames_dict)
        gm = gm.rename(self._renames)
        gm = self.add_indices(gm)
        return gm

    def rgba(self, xx: xr.Dataset) -> Optional[xr.DataArray]:
        if self.rgb_bands is None:
            return None
        return to_rgba(xx, clamp=self.rgb_clamp, bands=self.rgb_bands)


_plugins.register("crop_mask_feature_s2", CropMaskFeautures)
