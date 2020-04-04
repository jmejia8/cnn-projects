function init()
	for d in ["bson", "input_img", "outout_img"]
		!isdir(d) && mkdir(d)
	end
	

end

init()

