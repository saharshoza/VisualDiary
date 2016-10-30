import av

def frame_extractor(path_to_video,path_to_frames):
	container = av.open(path_to_video)
	video = next(s for s in container.streams if s.type == b'video')
	for packet in container.demux(video):
    	for frame in packet.decode():
        	frame.to_image().save(path_to_frames+'/frame-%04d.jpg' % frame.index)

if __name__ == "__main__":
	path_to_video = sys.argv[1]
	path_to_frames = sys.argv[2]