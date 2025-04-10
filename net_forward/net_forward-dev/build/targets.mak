TARGET_dir = output_object/$(CUST)/$(PROJECT)/$(TARGET)/
all_object_nordir = $(notdir $(objects))
all_object_dir = $(dir $(objects))
all_obj_pos = $(addprefix $(TARGET_dir),$(all_object_nordir))
# all_depend = $(all_obj_pos:.o=.d)
VPATH = $(all_object_dir)
all: $(all_obj_pos) #clean_depend
	- rm -vf $(TARGET).lib
	$(AR) $(LFLAGS) $(TARGET).lib $(all_obj_pos)

# $(all_depend):$(TARGET_dir)%.d:%.c
# 	@$(CC_depend) $(CFLAGS_depend)  -MM  $< > $@.1;\
# 	sed 's,^.*.o:,$(TARGET_dir)$*.o $(TARGET_dir)$*.d:,g' <$@.1> $@;\
# 	rm -f $@.1
#-lm
$(all_obj_pos):$(TARGET_dir)%.o:%.c $(depend_time)
	$(CC) -c $(CFLAGS) $(COMMON_CFLAGS) $(INCLUDES) $(COMMON_INCLUDES) $< -o $@

# -include $(all_depend)

.PHONY : clean new

# clean_depend:
# 	- rm -vf $(all_depend)
clean :
	- rm -vf *.lib $(TARGET_BASE)*.lib $(all_depend)

new : all
