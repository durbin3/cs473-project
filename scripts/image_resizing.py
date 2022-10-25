from PIL import ImageOps,Image
import os



def main():
    size = (1000,1000)
    directory = '../dataset/images/raw_images'
    out_dir = '../dataset/images/resized_images'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    for file in os.listdir(directory):
        if file.endswith(('jpeg', 'png', 'jpg')):
            outfile = os.path.join(out_dir, file)
            with Image.open(directory+'/'+file) as im:
                square = expand2square(im)
                out = square.resize(size,resample=Image.Resampling.LANCZOS)
                out.save(outfile,quality=100)

def expand2square(image):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), (255,255,255))
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), (255,255,255))
        result.paste(image, ((height - width) // 2, 0))
        return result


if __name__ == '__main__': main()