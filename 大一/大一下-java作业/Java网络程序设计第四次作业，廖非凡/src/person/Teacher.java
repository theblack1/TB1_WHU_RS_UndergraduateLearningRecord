package person;

import java.util.Vector;

import Course.Course;
import InterFaces.Stuff;

public class Teacher extends Ruler implements Stuff {
	String academy = "";
	Vector<Course> teachCo = new Vector<Course>();// 记录这个老师管理的课程

	public Teacher() {
	};

	// Teacher类创建自带等级；
	public Teacher(String name) {
		this.name = name;
		this.level = L4;
	}

	public Teacher(String name, String academy) {
		this.name = name;
		this.academy = academy;
	}

	public Teacher(String name, String academy, Course firstCourse) {
		this.name = name;
		this.academy = academy;

		firstCourse.boss = this;
		firstCourse.teacher = name;
		teachCo.add(firstCourse);

		this.level = L4;

		ShowMe();

	}

	// 添加课程
	public boolean Addcourse(Course course) {
		for (Course s : teachCo) {
			if (s == course) {
				return false;
			}
		}
		teachCo.add(course);
		return true;

	};

	// 上课
	public void teach(final Course course, int _showData) {// 输入非0参数默认显示课程详情

		Vector<Student> vStu = course.stuList;
		if (_showData != 0) {
			IsToken(course);
			System.out.println("开始上课！\nCourse:" + course.name + "\nTeacher:" + name + "\nStudents:");

			for (Student s : vStu) {
				// 检查到课情况
				System.out.print(s.name);
				if (!s.attend(course)) {
					System.out.println(" 旷课！");
					s.GetRecord(course).skipTime++;
				} // 旷课记录
				else
					System.out.println(" 已签到");
			}
		} else {
			for (Student s : vStu) {
				// 检查到课情况
				if (!s.attend(course))
					s.GetRecord(course).skipTime++;
			}
		}
		course._courseNum++;
	}

	public void teach(final Course course) {// 不输入参数默认不显示课程详情
		teach(course, 0);
	}

	// 判断是不是代课
	protected boolean IsToken(final Course course) {
		for (Course s : teachCo) {
			if (s == course)
				return false;
		}
		System.out.println("（本次为" + name + "代课）");
		return true;
	}

	// 布置作业
	public void assignHomework(Course course, int _showData2) {
		IsToken(course);
		Vector<Student> vStu = course.stuList;
		if (_showData2 != 0) {
			System.out.println("\n开始布置作业！\nCourse:" + course.name + "\nTeacher:" + name + "\nStudents:");
			for (Student s : vStu) {
				System.out.print(s.name);
				s.GetRecord(course).studyPower++;
				if (s.doHomework(course) == 1)
					System.out.println(" 未完成作业");
				else if (s.doHomework(course) == 2)
					System.out.println(" 超额完成作业");
				else
					System.out.println(" 正常完成作业");
			}
		} else {
			for (Student s : vStu) {
				s.GetRecord(course).studyPower++;
				s.doHomework(course);
			}
		}

	}

	public void assignHomework(final Course course) {// 不输入参数默认不显示课程详情
		assignHomework(course, 0);
	}

	// 考试
	public void exam(final Course course, int _showData) {
		if (_showData != 0) {
			IsToken(course);
			System.out.println("\n“" + course.name + "”期末考核开始！");
			Vector<Student> vStu = course.stuList;
			int _power;// 简化书写
			int _eScore;
			for (Student s : vStu) {
				_power = s.GetRecord(course).studyPower;
				_eScore = s.GetRecord(course).examScore = (50 + 50 * _power / (3 * course._courseNum));
				System.out.println(s.name + " get score:" + _eScore);
			}
		} else {
			Vector<Student> vStu = course.stuList;
			int _power;// 简化书写
			int _eScore;
			for (Student s : vStu) {
				_power = s.GetRecord(course).studyPower;
				_eScore = s.GetRecord(course).examScore = (50 + 50 * _power / (3 * course._courseNum));
			}
		}
	}

	public void exam(final Course course) {
		exam(course, 0);
	}

	// 评分
	public void grade(final Course course, int _showData3) {
		if (_showData3 != 0) {
			IsToken(course);
			System.out.println("\n“" + course.name + "”开始评分！");
			Vector<Student> vStu = course.stuList;
			int _Max = 0;
			int i = 0;

			for (Student s : vStu) {
				if (s.GetRecord(course).examScore == -1) {
					System.out.println(s.name + "没有参加期末考试，不能参与总评");
					continue;
				}
				int _score = s.GetRecord(course).score = (s.GetRecord(course).examScore * 70 / 100) + 30// 期末考试占比70%，平时考勤占比30%
						- s.GetRecord(course).skipTime * 30 / course._courseNum;

				if (vStu.elementAt(_Max).GetRecord(course).score < s.GetRecord(course).score)
					_Max = i;// 选择成绩最好的学生

				System.out.println(s.name + " score:" + _score);
				i++;
			}
			System.out.print("其中成绩最好的是：" + vStu.elementAt(_Max).name);// 显示成绩最好的学生
			if (course.monitor == vStu.elementAt(_Max))
				System.out.print("(班长)");
			System.out.println();
		} else {
			Vector<Student> vStu = course.stuList;

			for (Student s : vStu) {
				if (s.GetRecord(course).examScore == -1)
					continue;
				int _score = s.GetRecord(course).score = (s.GetRecord(course).examScore * 70 / 100) + 30// 期末考试占比70%，平时考勤占比30%
						- s.GetRecord(course).skipTime * 30 / course._courseNum;

			}

		}
	}

	public void grade(final Course course) {
		grade(course, 0);

	}

	public void ShowMe() {
		System.out.println();
		System.out.println(Ruler.L4 + ":\nname：" + name + "\nAcademy:" + academy);
		System.out.println("Course:");
		for (Course s : teachCo) {
			System.out.println("“" + s.name + "”");
		}
	}
}
