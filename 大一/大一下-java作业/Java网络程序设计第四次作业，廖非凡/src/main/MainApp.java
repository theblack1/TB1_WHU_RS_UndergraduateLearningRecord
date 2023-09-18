package main;

import java.util.Vector;

import Course.Course;
import Course.PE;
import Course.Practice;
import Course.Theory;
import InterFaces.Study;
import department.Academy;
import department.Admin;
import department.College;
import department.University;
import person.Overseas;
import person.Postgraduate;
import person.Ruler;
import person.Student;
import person.Teacher;
import person.Undergraduate;

public class MainApp {
	public static void main(String args[]) {

		// 【备注1】Course类就是"班级类”
		// 【备注2】有很多函数的实参表多了一个“1”这个是用来显示具体数据用的；
		// 初始化系统
		System.out.println("\n----------准备注册----------\n");
		// 学生：
		String[] newStudents = { "张三", "李四", "王五", "小黑", "小白" };
		Vector<Student> stuList = new Vector<Student>(5);

		for (int i = 0; i < 5; i++) {// 逐个填入
			Undergraduate stu1 = new Undergraduate(newStudents[i], "遥感信息工程学院");
			stuList.add(stu1);
		}

		// 非本科生
		Postgraduate p1 = new Postgraduate("刘学长", "遥感信息工程学院");
		Overseas o1 = new Overseas("Mr.Hulk", "遥感信息工程学院");
		stuList.add(o1);
		stuList.add(p1);// 研究生和留学生也选修java

		// 课程（班级）：
		Theory javaCo = new Theory("JAVA面向对象", 100, "熊老师", stuList.elementAt(2), stuList);
		Theory javaCo2 = new Theory("JAVA找不到对象", 100, "廖老师");
		Theory MathCo = new Theory("高等数学", 100, "王老师", stuList.elementAt(3), stuList);
		Theory EnglishCo = new Theory("大学英语", 100, "彭老师", stuList.elementAt(1), stuList);

		PE peCo = new PE("篮球入门", 100);
		peCo.AddStu(o1);
		peCo.AddStu(p1);
		peCo.monitor = p1;
		peCo.ShowMe();
		// 给非本科生的课
		Practice DeepCo = new Practice("深度学习", 100, "熊老师", p1, p1);
		Theory ChineseCo = new Theory("中文入门", 100, "彭老师", o1, o1);

		// 学生
		System.out.print("\n这里每类只展示一个学生：");
		stuList.elementAt(2).ShowMe();
		p1.ShowMe();
		o1.ShowMe();

		// 老师:
		Teacher javaTe = new Teacher("熊老师", "遥感信息工程学院");
		javaTe.Addcourse(javaCo);
		javaTe.Addcourse(DeepCo);
		Teacher javaTe2 = new Teacher("廖老师", "遥感信息工程学院");
		Teacher MathTe = new Teacher("王老师", "数学与统计学院", MathCo);
		Teacher EnglishTe = new Teacher("彭老师", "外国语学院");
		EnglishTe.Addcourse(ChineseCo);
		EnglishTe.Addcourse(EnglishCo);
		EnglishTe.ShowMe();
		Teacher peTe = new Teacher("马老师", "体育部");
		peTe.Addcourse(peCo);
		peCo.ShowMe();

		// 系主任
		Ruler XZhuRen = new Ruler(Ruler.L31, "奚主任");

		// 系
		Vector<Course> coList = new Vector<Course>();
		coList.add(javaCo);
		coList.add(javaCo2);
		coList.add(MathCo);
		coList.add(EnglishCo);
		coList.add(peCo);
		coList.add(ChineseCo);
		coList.add(DeepCo);

		College Xi = new College(College.T1, "遥感信息工程系", XZhuRen, coList);
		Xi.AddTeacher(javaTe2);
		Xi.AddTeacher(javaTe);
		Xi.ShowMe();

		// 实验室主任
		Ruler SZhuRen = new Ruler(Ruler.L32, "石主任");

		// 实验室
		College Shi = new College(College.T2, "遥感信息工程国家重点实验室", SZhuRen);

		// 研究所主任
		Ruler YZhuRen = new Ruler(Ruler.L32, "严主任");

		// 研究所
		College Shuo = new College(College.T3, "遥感研究所", YZhuRen);

		// 院长
		Ruler dean = new Ruler(Ruler.L21, "院院长");

		// 学院
		Academy academy = new Academy("遥感信息工程学院", dean, Xi, Shi, Shuo);
		Academy academy1 = new Academy("数学与统计学院");
		Academy academy2 = new Academy("外国语学院");
		academy1.ShowMe();
		academy2.ShowMe();

		// 行政主任
		Ruler ad = new Ruler(Ruler.L22, "教导主任");

		// 行政职员
		Ruler em = new Ruler(Ruler.L25, "廖职员");

		// 行政部门
		Admin admin = new Admin("教务处", ad, em);

		// 校长
		Ruler pr = new Ruler(Ruler.L1, "窦校长");

		// 大学
		Vector<Academy> acList = new Vector<Academy>();
		acList.add(academy);
		acList.add(academy1);
		acList.add(academy2);
		University WHU = new University("武汉大学", pr, acList, admin);

		System.out.println("\n----------注册完毕！准备开始上课！----------\n");
		System.out.println("\n【【【用接口简单实现多态如下】】】\n");

		Study current = (Study) stuList.elementAt(4);
		current.Study();
		current = p1;
		current.Study();
		current = o1;
		current.Study();

		System.out.println("\n【【【“授课”和“听课”的例子:】】】");
		MathTe.teach(javaCo, 1);// 加上后面这个数字可以显示课程详情信息，不加数字就不显示详细信息
		/*
		 * 授课用teach函数, 这个时候teach函数会调用学生的attend函数显示是否旷课,
		 * 如果旷课teacher就给student记录一次skipTime, 如果听课了student就加一个studyPower.
		 */
		System.out.println("\n【【【“布置作业”的例子】】】");
		javaTe.assignHomework(javaCo, 1);
		/*
		 * 每次用assignHomework函数布置作业，student就加一个studyPower
		 * 但是同学可能有人做得好，有人不做作业，对studyPower就有影响(体现在doHomework函数)
		 */

		// 为了更好地完成接下来的“考试”和“评分”，下面我们授课和布置作业多遍
		System.out.println("\n------经过一学期学习后------");
		for (int j = 2; j < 20; j++) {
			javaTe.teach(javaCo);
			javaTe.assignHomework(javaCo);

			MathTe.teach(MathCo);
			MathTe.assignHomework(MathCo);

			EnglishTe.teach(EnglishCo);
			EnglishTe.teach(EnglishCo);// 不妨一周上两次英语
			EnglishTe.assignHomework(EnglishCo);
		}

		System.out.println("\n【【【“期末考核”的例子】】】");
		javaTe.exam(javaCo, 1);

		System.out.println("\n（中途转入小亮，他没有java期末考成绩所以不算分）");
		Undergraduate s0 = new Undergraduate("小亮（转）", "数学与统计学院");
		javaCo.AddStu(s0);
		s0.Addcourse(EnglishCo);

		// 数学考试和英语考试
		MathTe.exam(MathCo);
		EnglishTe.exam(EnglishCo);

		System.out.println("\n【【【“期末评分”的例子】】】");
		javaTe.grade(javaCo, 1);
		MathTe.grade(MathCo);
		EnglishTe.grade(EnglishCo);

		System.out.println("\n【【【“输出某个学生的各门课程总成绩及平均分”的例子】】】\n");
		academy.GetScore(stuList.elementAt(3));
		academy.GetScore(s0);

		System.out.println("\n【【【“计算输出所有选修了Java课程的所有学生总成绩及平均分】】】\n");
		academy.GetScore(javaCo);
	}

}