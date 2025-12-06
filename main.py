from utils.preprocess import preprocess_image
from utils.slogans_generate import slogans_generate
from utils.poster_generate import poster_generate

image_path = 'example/full-boots-2.png'

image_feature = preprocess_image(image_path)

# #slogan
# slogan = sloge_generate(image_feature)

# #post
post = poster_generate(image_feature)

# #voice
# voice = voice_generate(image_slogan)

# #3D
# model_3d = model_3d_generate(image_feature)