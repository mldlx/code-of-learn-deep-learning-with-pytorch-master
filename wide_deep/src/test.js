/**
 * 修改人：harryhzhang
 * 修改时间：2018-11-23 00:21:03
 * 文件路径：source/fast/synthesize/ante/
 */

active = true;
name = "Riskutil_WXC2BCOMPLAINTSREALTIMEWARNINGPUSH";
zhName = "商业支付高频投诉商户实时预警推送";
ownerGroupId = SYNTHESIZE_GROUP_ID_EX_ANTE;
policyType = POLICY_TYPE_NONE;

ownerRoutes.push({
    "fn": routeFuncByCaseType,
    "args": [
        CASE_TYPE_WX_BUSINESS_COMPLAIN_NOTIFY
    ]
});

run = function () {

    var mapOutput,mapFixedEvent,objStandardData,objUser,objRecvUser,objUseUser,objTransaction,objCard,objRecvCard,objDevice,objIDCard,objMobile,objGuidDevice,objIMEIDevice,objMacDevice,objDiskDevice,objMerchant,objPool,objCommon,objSpecTrans,nEventId,isFusing;
    var _this = this,_context = _this.context;
    var arr6196Result7DaysCntStat = [] ;
    var arrMerchantTransList7 = [] ;
    var arrTransList = [] ;
    var ts_p = 0 ;
    var ts_cnt_bef = 0 ;
    var ts_cnt_today = 0 ;
    var nDate = "" ;
    var n7DaysCntAvg = 0 ;
    var famount_count = "" ;
    var ts_cnt_7day = 0 ;
    var nFirstAmtInt = 0 ;
    var ts_amt_today = 0 ;
    var arrLevel = [] ;
    var arr6143Result = [] ;
    var arr6196Result7Days = [] ;
    var ts_cnt_yd = 0 ;
    var fsp_amtcnt = "" ;
    var arrMerchantCompList8 = [] ;
    var arrMerchantCompList7 = [] ;
    var strLevel = "" ;
    var ts_cnt_qt = 0 ;
    var fcomplaint_detail = "" ;
    var arrComp7DayListAsc = [] ;
    var fsp_name = "" ;
    var fsp_createtime = "" ;
    var fmchid = "" ;
    var amtcnttsyd = [] ;
    var arr6196Result7DaysCnt = [] ;
    var n7DaysCntStd = 0 ;
    var fsp_flag = "" ;
    var arrComp7DayList = [] ;
    var ts_cnt_2day = 0 ;
    var fspid = "" ;
    var strMerchantTransList7 = "" ;
    var nMerchantCompList8Length = 0 ;
    var ts_cnt_fl = 0 ;
    var strMerchantCompList7 = "" ;
    var amtcnttsqt = [] ;
    var nFirstDate = 0 ;
    var str6196Result7Days = "" ;
    var nFirstCnt = 0 ;
    var fcomplain_list = "" ;
    var amtcnttsall7 = [] ;
    var ts_cnt_wj = 0 ;
    var arrTransListFirst = [] ;
    var amtcnttswj = [] ;
    var amtcnttsall1 = [] ;
    var amtcnttsjz = [] ;
    var amtcnttsall2 = [] ;
    var ts_cnt_jz = 0 ;
    var fsp_level = "" ;
    var fmchname = "" ;
    var strApplyTimeDate = "" ;
    var strFirstOrderDate = "" ;
    var amtcnttsfl = [] ;
    var strTransList = "" ;
    var strApplyTime = "" ;
    var comp7daybig = 0 ;
    var arrTag = [] ;
    var ts_amt_7day = 0 ;
    var diffDate = 0 ;
    var nFirstAmt = 0 ;
    var n_merchantinfo_applytime = 0 ;
    var str_merchant_day_trans_list = "" ;
    var str_merchant_30day_comp_all_list = "" ;
    var tNow = 0 ;
    var str_merchant_30day_comp_rebate_list = "" ;
    var str_merchant_30day_comp_parttime_list = "" ;
    var str_merchant_30day_comp_forbid_list = "" ;
    var str_merchant_30day_comp_temptation_list = "" ;
    var str_merchant_30day_comp_other_list = "" ;
    var str_merchant_mch_id = "" ;
    var str_merchant_mch_id_name = "" ;
    var str_complaint_detail = "" ;
    var n_merchant_mch_model = 0 ;
    var str_merchantinfo_spid = "" ;
    var str_merchantinfo_name = "" ;
    var str_merchant_sub_mchid = "" ;
    var str_merchant_sub_mchid_name = "" ;
    var str_merchant_level_for_weixin = "" ;
    var str_qq = "" ;
    var str_listid = "" ;
    var str_sale_qq = "" ;
    var _n_REF_LGC_b2a422da244d;
    var _n_REF_LGC_33486cd160d8;
    var _n_REF_LGC_ef08151a55dc;
    var _n_REF_LGC_37f7257e9498;
    var str_merchant_first_order_amount_ten_info = "" ;
    var str_merchant_first_order_info = "" ;
    var strSQLAuto;
    var nSQLRetAuto;
    tNow = _context.tNow;


    mapOutput = _context.mapOutput;
    mapFixedEvent = _context.mapFixedEvent;
    objUser = _context.objUser;
    objRecvUser = _context.objRecvUser;
    objUseUser = _context.objUseUser;
    objTransaction = _context.objTransaction;
    objCard = _context.objCard;
    objRecvCard = _context.objRecvCard;
    objDevice = _context.objDevice;
    objIDCard = _context.objIDCard;
    objMobile = _context.objMobile;
    objGuidDevice = _context.objGuidDevice;
    objIMEIDevice = _context.objIMEIDevice;
    objMacDevice = _context.objMacDevice;
    objDiskDevice = _context.objDiskDevice;
    objPool = _context.objPool;
    objMerchant = _context.objMerchant;
    objCommon = _context.objCommon;
    objSpecTrans = _context.objSpecTrans;
    isFusing = _context.isFusing;//决策结果是否命中熔断，true表示命中熔断  false表示没有命中熔断
    objStandardData = mapFixedEvent["data"];
    nEventId = mapFixedEvent["event_id"];


    str_qq = MyParseString(objStandardData[QQ]);
    str_listid = MyParseString(objStandardData[LISTID]);
    str_complaint_detail = MyParseString(objStandardData[COMPLAINT_DETAIL]);
    str_sale_qq = MyParseString(objStandardData[SALE_QQ]);

    if ("undefined" != typeof(objMerchant))
    {
        str_merchant_level_for_weixin = MyParseString(objMerchant[MERCHANT_LEVEL_FOR_WEIXIN]);
        n_merchant_mch_model = MyParseInt(objMerchant[MERCHANT_MCH_MODEL]);
        str_merchant_30day_comp_rebate_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_REBATE_LIST]);
        str_merchant_first_order_info = MyParseString(objMerchant[MERCHANT_FIRST_ORDER_INFO]);
        str_merchant_mch_id = MyParseString(objMerchant[MERCHANT_MCH_ID]);
        str_merchant_30day_comp_all_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_ALL_LIST]);
        n_merchantinfo_applytime = MyParseInt(objMerchant[MERCHANTINFO_APPLYTIME]);
        str_merchant_30day_comp_temptation_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_TEMPTATION_LIST]);
        str_merchant_30day_comp_forbid_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_FORBID_LIST]);
        str_merchantinfo_name = MyParseString(objMerchant[MERCHANTINFO_NAME]);
        str_merchant_30day_comp_other_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_OTHER_LIST]);
        str_merchant_mch_id_name = MyParseString(objMerchant[MERCHANT_MCH_ID_NAME]);
        str_merchant_day_trans_list = MyParseString(objMerchant[MERCHANT_DAY_TRANS_LIST]);
        str_merchant_30day_comp_parttime_list = MyParseString(objMerchant[MERCHANT_30DAY_COMP_PARTTIME_LIST]);
        str_merchant_sub_mchid_name = MyParseString(objMerchant[MERCHANT_SUB_MCHID_NAME]);
        str_merchant_first_order_amount_ten_info = MyParseString(objMerchant[MERCHANT_FIRST_ORDER_AMOUNT_TEN_INFO]);
        str_merchantinfo_spid = MyParseString(objMerchant[MERCHANTINFO_SPID]);
        str_merchant_sub_mchid = MyParseString(objMerchant[MERCHANT_SUB_MCHID]);
    }

    do{

        mapOutput["data"] = {};
        arrTag = MyParseArray(str_merchant_first_order_amount_ten_info.split(","));
        arr6143Result = MyParseArray(str_merchant_first_order_info.split(","));
        arrLevel = MyParseArray(str_merchant_level_for_weixin.split(","));
        arrTransList = MyParseArray(str_merchant_day_trans_list.split(";"));

        nDate = MyParseString(MyParseInt(arrTag[0]));
        strApplyTime = MyParseString(GetStringFromTime(n_merchantinfo_applytime));
        arrMerchantTransList7 = MyParseArray(getDateList(str_merchant_day_trans_list,7));
        arrMerchantCompList7 = MyParseArray(getDateList(str_merchant_30day_comp_all_list,7));
        arrMerchantCompList8 = MyParseArray(getDateList(str_merchant_30day_comp_all_list,8));

        diffDate = MyParseInt(getDeltaByDate(tNow,0,nDate,2,1));
        amtcnttsall1 = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_all_list,1,""));
        amtcnttsall2 = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_all_list,2,""));
        amtcnttsall7 = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_all_list,7,""));
        amtcnttsfl = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_rebate_list,7,""));
        amtcnttsjz = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_parttime_list,7,""));
        amtcnttswj = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_forbid_list,7,""));
        amtcnttsyd = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_temptation_list,7,""));
        amtcnttsqt = MyParseArray(GetRecentDaysAmountAndCount(str_merchant_30day_comp_other_list,7,""));

        strMerchantCompList7 = MyParseString(arrMerchantCompList7.join(";"));
        strMerchantTransList7 = MyParseString(arrMerchantTransList7.join(";"));
        strApplyTimeDate = MyParseString(strApplyTime.substring(0,10));
        nMerchantCompList8Length = MyParseInt(arrMerchantCompList8.length);
        arr6196Result7Days = MyParseArray(arrMerchantCompList8.slice(1,nMerchantCompList8Length));
        str6196Result7Days = MyParseString(arr6196Result7Days.join(";"));

        if(nMerchantCompList8Length > 1){
            arr6196Result7DaysCnt = MyParseArray(getColumnArrayByList(str6196Result7Days));
            arr6196Result7DaysCntStat = MyParseArray(getDataMiningValues(arr6196Result7DaysCnt[2]));
            n7DaysCntAvg = (arr6196Result7DaysCntStat[0]);
            n7DaysCntStd = (arr6196Result7DaysCntStat[4]);
            _n_REF_LGC_b2a422da244d = true;
        }else{
            _n_REF_LGC_b2a422da244d = true;
        }

        if(_n_REF_LGC_b2a422da244d){
            ts_amt_today = MyParseInt(amtcnttsall1[0]);
            ts_cnt_today = MyParseInt(amtcnttsall1[1]);
            ts_amt_7day = MyParseInt(amtcnttsall7[0]);
            ts_cnt_7day = MyParseInt(amtcnttsall7[1]);
            ts_cnt_fl = MyParseInt(amtcnttsfl[1]);
            ts_cnt_jz = MyParseInt(amtcnttsjz[1]);
            ts_cnt_wj = MyParseInt(amtcnttswj[1]);
            ts_cnt_yd = MyParseInt(amtcnttsyd[1]);
            ts_cnt_qt = MyParseInt(amtcnttsqt[1]);
            strLevel = MyParseString(arrLevel[0]);
            strFirstOrderDate = MyParseString(arr6143Result[0]);
            ts_cnt_2day = MyParseInt(amtcnttsall2[1]);
            strTransList = MyParseString(arrTransList[0]);
            arrTransListFirst = MyParseArray(strTransList.split(","));
            ts_p = ((ts_cnt_fl + ts_cnt_jz + ts_cnt_wj + ts_cnt_yd + ts_cnt_qt) / ts_cnt_7day);
            arrComp7DayList[0] = ts_cnt_fl;
            arrComp7DayList[1] = ts_cnt_jz;
            arrComp7DayList[2] = ts_cnt_yd;
            ts_cnt_bef = MyParseInt(ts_cnt_2day - ts_cnt_today);
            nFirstDate = MyParseInt(arrTransListFirst[0]);
            nFirstAmt = (arrTransListFirst[1] / 100);
            nFirstCnt = MyParseInt(arrTransListFirst[2]);
            nFirstAmtInt = MyParseInt(nFirstAmt.toFixed(0));
            if(strApplyTimeDate == "1970-01-01"){
                fsp_createtime = MyParseString(strFirstOrderDate);
                _n_REF_LGC_33486cd160d8 = true;
            }else{
                fsp_createtime = MyParseString(strApplyTimeDate);
                _n_REF_LGC_33486cd160d8 = true;
            }
        }

        if(_n_REF_LGC_33486cd160d8){
            fmchid = MyParseString(str_merchant_mch_id);
            fmchname = MyParseString(str_merchant_mch_id_name);
            fsp_level = MyParseString(strLevel);
            fsp_amtcnt = MyParseString(strMerchantTransList7);
            fcomplain_list = MyParseString(strMerchantCompList7);
            fcomplaint_detail = MyParseString(str_complaint_detail);
            famount_count = MyParseString("日期:" + nFirstDate + ";金额:" + nFirstAmtInt + "元;笔数:" + nFirstCnt);
            arrComp7DayListAsc = MyParseArray(sortNumberArray(arrComp7DayList,0));
            comp7daybig = MyParseInt(arrComp7DayListAsc[2]);
            if(comp7daybig == ts_cnt_fl){
                fsp_flag = "TOP投诉-返利";
                _n_REF_LGC_ef08151a55dc = true;
            }
            if(comp7daybig == ts_cnt_jz){
                fsp_flag = "TOP投诉-兼职";
                _n_REF_LGC_ef08151a55dc = true;
            }
            if(comp7daybig == ts_cnt_yd){
                fsp_flag = "TOP投诉-诱导";
                _n_REF_LGC_ef08151a55dc = true;
            }
        }

        if(_n_REF_LGC_ef08151a55dc){
            if(n_merchant_mch_model == 0 || n_merchant_mch_model == 1){
                fspid = MyParseString(str_merchantinfo_spid);
                fsp_name = MyParseString(str_merchantinfo_name);
                _n_REF_LGC_37f7257e9498 = true;
            }else{
                fspid = MyParseString(str_merchant_sub_mchid);
                fsp_name = MyParseString(str_merchant_sub_mchid_name);
                _n_REF_LGC_37f7257e9498 = true;
            }
        }

        if(_n_REF_LGC_37f7257e9498){
            if((tNow - n_merchantinfo_applytime <= 365 * 24 * 3600 || diffDate <= 365) && MyParseString(str_merchant_level_for_weixin).indexOf(MyParseString("A")) == -1 && MyParseString(str_merchant_level_for_weixin).indexOf(MyParseString("S")) == -1 && ((ts_cnt_today >= 20 && (ts_amt_today >= 2000 * 100 || (ts_cnt_today >= n7DaysCntAvg + 3 * n7DaysCntStd && n7DaysCntAvg > 0 && n7DaysCntStd > 0) || (ts_cnt_today >= 1.75 * ts_cnt_bef && ts_cnt_bef > 0))) || ts_cnt_today >= 30) && ts_p >= 0.7 && str_complaint_detail != ""){
                strSQLAuto = "insert into t_c2b_evilsp_push set Fspid = '" + fspid + "' , Fsp_name = '" + fsp_name + "' , Fmchid = '" + fmchid + "' , Fmchname = '" + fmchname + "' , Fsp_level = '" + fsp_level + "' , Fsp_flag = '" + fsp_flag + "' , Fsp_amtcnt = '" + fsp_amtcnt + "' , Fcomplain_list = '" + fcomplain_list + "' , Fstatus = 0 , Ftype = 1 , Fsp_createtime = '" + fsp_createtime + "' , Fmodify_time = now() , Fcomplaint_detail = '" + fcomplaint_detail + "' , Famount_count = '" + famount_count + "';";
                nSQLRetAuto = DB_Execute(strSQLAuto, SQL_MODE_ASYNC);

            }
        }

    }while(false);

    DebugPrintByLevel_JS("WXC2BCOMPLAINTSREALTIMEWARNINGPUSH:" + "1="+nEventId+ ", 2="+str_qq+ ", 3="+str_sale_qq+ ", 4="+str_listid+ ", 5="+strSQLAuto+ ", 6="+nSQLRetAuto, LOG_LEVEL_DEBUG);
    DebugPrintByLevel_JS("WXC2BCOMPLAINTSREALTIMEWARNINGPUSH:" + "1="+nEventId+ ", 2="+str_qq+ ", 3="+str_sale_qq+ ", 4="+str_listid+ ", 5="+arr6196Result7DaysCntStat+ ", 6="+arrTransList+ ", 7="+arrMerchantTransList7+ ", 8="+ts_p+ ", 9="+str_complaint_detail+ ", 10="+ts_cnt_bef+ ", 11="+ts_cnt_today+ ", 12="+nDate+ ", 13="+n7DaysCntAvg+ ", 14="+famount_count+ ", 15="+ts_cnt_7day+ ", 16="+nFirstAmtInt+ ", 17="+ts_amt_today+ ", 18="+arrLevel+ ", 19="+arr6143Result+ ", 20="+arr6196Result7Days+ ", 21="+ts_cnt_yd+ ", 22="+arrMerchantCompList8+ ", 23="+fsp_amtcnt+ ", 24="+arrMerchantCompList7+ ", 25="+strLevel+ ", 26="+ts_cnt_qt+ ", 27="+fcomplaint_detail+ ", 28="+arrComp7DayListAsc+ ", 29="+fsp_name+ ", 30="+fsp_createtime+ ", 31="+fmchid+ ", 32="+amtcnttsyd+ ", 33="+arr6196Result7DaysCnt+ ", 34="+n7DaysCntStd+ ", 35="+fsp_flag+ ", 36="+ts_cnt_2day+ ", 37="+n_merchant_mch_model+ ", 38="+fspid+ ", 39="+strMerchantTransList7+ ", 40="+nMerchantCompList8Length+ ", 41="+str_merchant_level_for_weixin+ ", 42="+ts_cnt_fl+ ", 43="+strMerchantCompList7+ ", 44="+amtcnttsqt+ ", 45="+nFirstDate+ ", 46="+str6196Result7Days+ ", 47="+nFirstCnt+ ", 48="+fcomplain_list+ ", 49="+amtcnttsall7+ ", 50="+ts_cnt_wj+ ", 51="+arrTransListFirst+ ", 52="+amtcnttswj+ ", 53="+amtcnttsall1+ ", 54="+amtcnttsjz+ ", 55="+amtcnttsall2+ ", 56="+ts_cnt_jz+ ", 57="+fsp_level+ ", 58="+fmchname+ ", 59="+strApplyTimeDate+ ", 60="+strFirstOrderDate+ ", 61="+amtcnttsfl+ ", 62="+strTransList+ ", 63="+strApplyTime+ ", 64="+comp7daybig+ ", 65="+tNow+ ", 66="+arrTag+ ", 67="+ts_amt_7day+ ", 68="+n_merchantinfo_applytime+ ", 69="+diffDate+ ", 70="+nFirstAmt, LOG_LEVEL_INFO);
}

//事件监听部分



