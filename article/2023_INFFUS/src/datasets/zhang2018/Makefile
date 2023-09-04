pretrained := 1kHJUqb-e7BARb63741DVdpg-WqCdG3z6

Confocal_BPAE_B := 1juaumcGn5QlFRXRQyrqfbZBhF7oX__iW
Confocal_BPAE_G := 1Zofz11VmI1JfRIMF7rq40RVjpzM6A9vg
Confocal_BPAE_R := 1QoD_vMvFdFg7yREfen3t-SGLFcnLg9YQ
Confocal_FISH := 1SxmsythWfxnfKJfGWpT_7Adebi8jUK98
Confocal_MICE := 11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ

TwoPhoton_BPAE_B := 1yVD_H_ZfNNSma5vtHZM_DTnSv1Bo1tfk
TwoPhoton_BPAE_G := 125nqTfQQG1-YVUs256b2vTwt4aUNCgBt
TwoPhoton_BPAE_R := 1rwxG6LYcKeiBKNT3Oq9lvwKu8mV3rz9P
TwoPhoton_MICE := 1lhsFAlXsXk26yqHzT0_-3R8MUb7G0NVa

WideField_BPAE_B := 19rl8zFzfXIZ2drgodCGutLPLzL4kJq6d
WideField_BPAE_G := 1H67O6GqIkIlQSX-n0vfMWGPwmd4zOHQr
WideField_BPAE_R := 19HXb2Ftrb-M7Lr9ZlHWMcnNT0Sbu85YL

DATASETS := Confocal_BPAE_B \
	Confocal_BPAE_G \
	Confocal_BPAE_R \
	Confocal_FISH \
	Confocal_MICE \
	TwoPhoton_BPAE_B \
	TwoPhoton_BPAE_G \
	TwoPhoton_BPAE_R \
	TwoPhoton_MICE \
	WideField_BPAE_B \
	WideField_BPAE_G \
	WideField_BPAE_R \

all: models pretrained Confocal_BPAE_B

$(DATASETS):
	wget "https://docs.google.com/uc?export=download&id=$($@)&confirm=yes" -O $@.tar
	tar -xf $@.tar
	rm $@.tar
	rm -r $@/avg* $@/gt

pretrained:
	wget "https://docs.google.com/uc?export=download&id=$($@)&confirm=yes" -O $@.tar
	tar -xf $@.tar
	rm $@.tar

models:
	wget -O $@.zip https://github.com/yinhaoz/denoising-fluorescence/archive/d83cad96b205793da0f62f6dc8094799d61929e6.zip
	unzip $@.zip "denoising-fluorescence-d83cad96b205793da0f62f6dc8094799d61929e6/denoising/models/*"
	mv denoising-fluorescence-d83cad96b205793da0f62f6dc8094799d61929e6/denoising/models/ .
	rm -r denoising-fluorescence-d83cad96b205793da0f62f6dc8094799d61929e6/
	rm $@.zip
