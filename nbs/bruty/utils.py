import numpy
from osgeo import gdal, osr, ogr

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterate_stuff, *args, **kywrds):
        return iterate_stuff  # if this doesn't work, try iter(iterate_stuff)


def affine(r, c, x0, dxx, dyx, y0, dxy, dyy):
    """
    Returns the affine transform -- normally row, column to x,y position.
    If this is the geotransform from a gdal geotiff (for example) the coordinates are the displayed pixel corners - not the center.
    If you want the center of the pixel then use affine_center
    """
    x = x0 + c * dxx + r * dyx
    y = y0 + c * dxy + r * dyy
    return x, y


def inv_affine(x, y, x0, dxx, dyx, y0, dxy, dyy):
    if dyx == 0 and dxy == 0:
        c = numpy.array(numpy.floor((numpy.array(x) - x0) / dxx), dtype=numpy.int32)
        r = numpy.array(numpy.floor((numpy.array(y) - y0) / dyy), dtype=numpy.int32)
    else:
        # @todo support skew projection
        raise ValueError("non-North up affine transforms are not supported yet")
    return r, c


def affine_center(r, c, x0, dxx, dyx, y0, dxy, dyy):
    return affine(r + 0.5, c + 0.5, x0, dxx, dyx, y0, dxy, dyy)


def merge_arrays(x, y, keys, data, *args, **kywrds):
    # create a combined two dimensional array that can then be indexed and reduced as needed,
    #   if the incoming data was multidimensional this will flatten it too.
    #   So an incoming dataset is say 6 fields of 4 rows and 5 columns will be turned into 10 x 20 -- (6+4 fields x 4 rows * 5 cols)
    #   wouldn't be able to remove nans and potentially do some other operations otherwise(?).
    pts = numpy.array((x, y, *keys, *data)).reshape(2 + len(keys) + len(data), -1)
    merge_array(pts, *args, **kywrds)


def merge_array(pts, output_data, output_sort_key_values,
                crs_transform=None, affine_transform=None, start_col=0, start_row=0, block_cols=None, block_rows=None,
                reverse_sort=None, key_bounds=None
                ):
    """  Merge a new dataset (array) into an existing dataset (array).
    The source data is compared to the existing data to determine if it should supercede the existing data.
    For example, this could be hydro-health score comparison for higher quality data or depths for shoal biasing.

    The incoming dataset uses a geotransform to go from source to destination coordinates
    then an affine transform to go from destination x,y to array row, column

    Basically pass in x,y,(sort_key1, sort_key2, ...), data_to_for_output_array, output_array, output_sort_key_result
    if affine_transform is None then pass in row, column instead of x,y

    note: modifies the output array in place

    Parameters
    ----------
    pts
        x,y, sort_key1, sort_key2, ..., data_for_output_array
        The number of sort keys is defined by the length of the supplied output_sort_key_values array.
    output_data
        array of length that matches the 'data_for_output_array' without desired rows/cols for the data to be inserted to.
        Must match the number of arrays passed in after the sort_keys
    output_sort_key_values
        array of length must match the number of sort keys passed in.
        Must match the row/column size of the output_data
    crs_transform
        object with a .transform(x,y) method returning x2, y2
    affine_transform
        If supplied, converts from x,y to row,col
    start_col
        column offset value for the output_array.
    start_row
        row offset value for the output_array.
    block_cols
        maximum column to fill with data
    block_rows
        maximum row to fill with data
    reverse_sort
        If supplied, an iterable of booleans specifying if the corresponding sort_key is reversed.
        ex: if sort keys were (z,x,y) and the smallest z was desired then (True, False, False) would be the reverse_sort value
    key_bounds
        Ranges of acceptable values for each sort passed in as (min, max).  Values outside this range will be discarded.
        None can be supplied if no min or max is needed.  Either min or max could also be None.

        ex: given a (z,lat,lon) sort, if elevations were desired between 0m and 40m and only from 30deg to 40deg latitude you
        would specify ((0, 40), (30, 40), None).

        ex: given a (z,lat,lon) sort, if elevations were desired below 0m and above 40deg latitude you
        would specify ((None, 0), (40, None), None).

        !!Note - exclusive bands do not work!!  passing in (40, 0) to try and get above 40 or below 0 will return nothing.
        @todo add the ability to have a callback or specify the if the range is and/or so excludes would work.

    Returns
    -------
    None
    """

    pts = pts[:, ~numpy.isnan(pts[2])]  # remove empty cells (no score = empty)
    if len(pts[0]) > 0:
        if block_rows is None:
            row_index = 1 if len(output_data.shape) > 2 else 0
            block_rows = output_data.shape[row_index] - start_row
        if block_cols is None:
            col_index = 2 if len(output_data.shape) > 2 else 1
            block_cols = output_data.shape[col_index] - start_col

        # 6) Sort on score in case multiple points go into a position that the right value is retained
        #   sort based on score then on depth so the shoalest top score is kept
        if reverse_sort is None:
            sort_multiplier = [1] * len(output_sort_key_values)
        else:
            sort_multiplier = [-1 if flag else 1 for flag in reverse_sort]

        # sort the points, the following puts all the keys in backwards (how lexsort wants) and flips signs as needed
        sorted_ind = numpy.lexsort([sort_multiplier[num_key] * pts[num_key + 2] for num_key in range(len(output_sort_key_values) - 1, -1, -1)])
        # sorted_ind = numpy.lexsort((sort_z_multiplier * pts[3], sort_score_multiplier * pts[2]))
        sorted_pts = pts[:, sorted_ind]

        # 7) Use affine geotransform convert x,y into the i,j for the exported area
        if crs_transform:
            transformed_x, transformed_y = crs_transform.transform(sorted_pts[0], sorted_pts[1])
        else:
            transformed_x, transformed_y = sorted_pts[0], sorted_pts[1]

        if affine_transform is not None:
            export_rows, export_cols = inv_affine(transformed_x, transformed_y, *affine_transform)
        else:
            export_rows, export_cols = transformed_x.astype(numpy.int32), transformed_y.astype(numpy.int32)
        export_rows -= start_row  # adjust to the sub area in memory
        export_cols -= start_col

        # clip to the edges of the export area since our db tiles can cover the earth [0:block_rows-1, 0:block_cols]
        row_out_of_bounds = numpy.logical_or(export_rows < 0, export_rows >= block_rows)
        col_out_of_bounds = numpy.logical_or(export_cols < 0, export_cols >= block_cols)
        out_of_bounds = numpy.logical_or(row_out_of_bounds, col_out_of_bounds)
        if out_of_bounds.any():
            sorted_pts = sorted_pts[:, ~out_of_bounds]
            export_rows = export_rows[~out_of_bounds]
            export_cols = export_cols[~out_of_bounds]

        # 8) Write the data into the export (single) tif.
        # replace x,y with row, col for the points
        # @todo write unit test to confirm that the sort is working in case numpy changes behavior.
        #   currently assumes the last value is stored in the array if more than one have the same ri, rj indices.
        replace_cells = numpy.isnan(output_sort_key_values[0, export_rows, export_cols])
        previous_all_equal = numpy.full(replace_cells.shape,
                                        True)  # tracks if the sort keys are all equal in which case we have to check the next key
        key_in_bounds = numpy.full(replace_cells.shape, True)
        for key_num, key in enumerate(output_sort_key_values):
            if reverse_sort is None or not reverse_sort[key_num]:
                comp_func = numpy.greater
            else:
                comp_func = numpy.less
            # in the compare function use astype to cast the incoming data to the same type as the exiting data.
            # this handles problems of if the new depths are float64 but the stored data is float32 that rounding (representation) errors will occur
            replace_cells = numpy.logical_or(replace_cells,
                                             numpy.logical_and(previous_all_equal,
                                                               comp_func(sorted_pts[2 + key_num].astype(key.dtype), key[export_rows, export_cols])))
            previous_all_equal = numpy.logical_and(previous_all_equal,
                                                   numpy.equal(sorted_pts[2 + key_num], key[export_rows, export_cols]))
            # check that the key values are within the desired ranges
            if key_bounds is not None:
                if key_bounds[key_num] is not None:
                    key_min, key_max = key_bounds[key_num]
                    if key_min is not None:
                        key_in_bounds = numpy.logical_and(key_in_bounds, sorted_pts[2 + key_num] >= key_min)
                    if key_max is not None:
                        key_in_bounds = numpy.logical_and(key_in_bounds, sorted_pts[2 + key_num] <= key_max)
        replace_cells = numpy.logical_and(replace_cells, key_in_bounds)

        replacements = sorted_pts[2 + len(output_sort_key_values):, replace_cells]
        ri = export_rows[replace_cells]
        rj = export_cols[replace_cells]
        output_data[:, ri, rj] = replacements
        for key_num, key in enumerate(output_sort_key_values):
            key[ri, rj] = sorted_pts[2 + key_num, replace_cells]


def soundings_from_image(fname, res):
    ds = gdal.Open(str(fname))
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    affine_params = ds.GetGeoTransform()
    xform = ds.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
    d_val = ds.GetRasterBand(1)
    col_size = d_val.XSize
    row_size = d_val.YSize
    del d_val
    x1, y1 = affine(0, 0, *xform)
    x2, y2 = affine(row_size, col_size, *xform)
    try:  # allow res to be tuple or single value
        res_x, res_y = res
    except TypeError:
        res_x = res_y = res

    # move the minimum to an origin based on the resolution so future exports would match
    # ex: res = 50 would make origin be 0 or 50 or 100 but not contain 77.3 etc.
    if x1 < x2:
        x1 -= x1 % res_x
    else:
        x2 -= x2 % res_x

    if y1 < y2:
        y1 -= y1 % res_y
    else:
        y2 -= y2 % res_y
    min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
    # create an x,y,z array of nans in the necessary shape
    output_array = numpy.full([3, shape_y, shape_x], numpy.nan, dtype=numpy.float64)
    output_sort_values = numpy.full([3, shape_y, shape_x], numpy.nan, dtype=numpy.float64)
    output_xform = [min_x, res_x, 0, max_y, 0, -res_y]
    layers = [ds.GetRasterBand(b + 1).GetDescription().lower() for b in range(ds.RasterCount)]
    band_num = None
    for name in ('elevation', 'depth'):
        if name in layers:
            band_num = layers.index(name) + 1
            break
    if band_num is None:
        band_num = 1
    for ic, ir, nodata, (depths,) in iterate_gdal_image(ds, (band_num,)):
        depths[depths == nodata] = numpy.nan
        r, c = numpy.indices(depths.shape)
        x, y = affine_center(r + ir, c + ic, *xform)
        # first key is depth second is latitude (just to have a tiebreaker so results stay consistent)
        # reusing the depth (output_array[2]) and y output_array[1] in the sortkey arrays
        merge_arrays(x, y, (depths, x, y), (x, y, depths), output_array, output_sort_values, affine_transform=output_xform)
    return srs, output_array


def iterate_gdal_image(dataset, band_nums=(1,), min_block_size=512, max_block_size=1024,
                       start_col=0, end_col=None, start_row=0, end_row=None, leave_progress_bar=True):
    """ Iterate a gdal dataset using blocks to reduce memory usage.
    The last blocks at the edge of the dataset will have a smaller size than the others.
    Reads down all the rows first then moves to the next group of columns.
    The function will use the dataset's GetBlockSize() if it falls between the min and max block size arguments

    Ex: a 10x10 image read in blocks of 4x4 would return  two 4x4 arrays followed by a 2x4 array.  Then 4x4, 4x4, 2x4.  Then 4x2, 4x2, 2x2.

    Parameters
    ----------
    dataset
        gdal dataset to read from
    band_nums
        list of band number integers to read arrays from
    min_block_size
        minimum size to allow block reads to use
    max_block_size
        maximum size to allow block reads to use

    Returns
    -------
    ic, ir, nodata, data
        column index, row index, no data value, list of arrays from the dataset

    """
    bands = [dataset.GetRasterBand(num) for num in band_nums]
    block_sizes = bands[0].GetBlockSize()
    row_block_size = min(max(block_sizes[1], min_block_size), max_block_size)
    col_block_size = min(max(block_sizes[0], min_block_size), max_block_size)
    col_size = bands[0].XSize
    if end_col is not None and end_col >= 0 and end_col <= col_size:
        col_size = end_col
    row_size = bands[0].YSize
    if end_row is not None and end_row >= 0 and end_row <= row_size:
        row_size = end_row
    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0
    nodata = bands[0].GetNoDataValue()
    # read the data array in blocks
    for ic in tqdm(range(start_col, col_size, col_block_size), desc='column block', mininterval=.7, leave=leave_progress_bar):
        if ic + col_block_size < col_size:
            cols = col_block_size
        else:
            cols = col_size - ic
        for ir in tqdm(range(start_row, row_size, row_block_size), desc='row block', mininterval=.7, leave=False):
            if ir + row_block_size < row_size:
                rows = row_block_size
            else:
                rows = row_size - ir
            yield ic, ir, nodata, [band.ReadAsArray(ic, ir, cols, rows) for band in bands]


def save_soundings_from_image(inputname, outputname, res, flip_depth=True):
    srs, sounding_array = soundings_from_image(inputname, res)
    # make a geopackage of the x,y,z values held in the sounding matrix
    sounding_array = sounding_array.reshape(sounding_array.shape[0], -1)
    sounding_array = sounding_array[:, ~numpy.isnan(sounding_array[2])]
    if flip_depth:
        sounding_array[2] *= -1

    dst_ds = ogr.GetDriverByName('Memory').CreateDataSource(outputname)
    lyr = dst_ds.CreateLayer('SOUNDG', srs, ogr.wkbPoint)

    # match the geopackage format from Caris
    for field in (
            ogr.FieldDefn('SOUACC', ogr.OFTReal),
            ogr.FieldDefn('SORDAT', ogr.OFTString),
            ogr.FieldDefn('SORIND', ogr.OFTString),
    ):
        if 0 != lyr.CreateField(field):
            raise RuntimeError("Creating field failed.", field.GetName())

    sounding_array = sounding_array.astype(numpy.float64).T
    for x, y, z in sounding_array:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), float(y), float(z))
        # Create a feature, using the attributes/fields that are required for this layer
        feat = ogr.Feature(feature_def=lyr.GetLayerDefn())
        feat.SetGeometry(point)
        lyr.CreateFeature(feat)
        # Clean up
        feat.Destroy()
    ogr.GetDriverByName('GPKG').CopyDataSource(dst_ds, dst_ds.GetName())


def calc_area_array_params(x1, y1, x2, y2, res_x, res_y, align_x=None, align_y=None):
    """ Compute a coherent min and max position and shape given a resolution.
    Basically we may know the desired corners but they won't fit perfectly based on the resolution.
    So we will compute what the adjusted corners would be and number of cells needed based on the resolution provided.

    ex: (0, 10, 5, 0, 4, 3) would return (0, 0, 8, 12, 2, 4)
    The minimum is held constant (0,0) the other corner would be moved from (5, 10) to (8, 12) because the resolution was (4,3)
    and there would be 2 columns (xsize) and 4 rows (ysize)

    Parameters
    ----------
    x1
        an X corner coordinate
    y1
        an Y corner coordinate
    x2
        an X corner coordinate
    y2
        an Y corner coordinate
    res_x
        pixel size in x direction
    res_y
        pixel size in y direction
    align_x
        if supplied the min_x will be shifted to align to an integer cell offset from the align_x, if None then no effect
    align_y
        if supplied the min_y will be shifted to align to an integer cell offset from the align_y, if None then no effect
    Returns
    -------
    min_x, min_y, max_x, max_y, cols (shape_x), rows (shape_y)

    """
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    if align_x:
        min_x -= (min_x - align_x) % res_x
    if align_y:
        min_y -= (min_y - align_y) % res_y
    shape_x = int(numpy.ceil((max_x - min_x) / res_x))
    shape_y = int(numpy.ceil((max_y - min_y) / res_y))
    max_x = shape_x * res_x + min_x
    max_y = shape_y * res_y + min_y
    return min_x, min_y, max_x, max_y, shape_x, shape_y
